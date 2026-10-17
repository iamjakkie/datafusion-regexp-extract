use datafusion::arrow::array::{AsArray, Int64Array, StringArray};
use datafusion::arrow::datatypes::{DataType, Field, Schema};
use datafusion::arrow::record_batch::RecordBatch;
use datafusion::error::Result;
use datafusion::prelude::{SessionContext, col, lit};
use datafusion_common::ScalarValue;
use datafusion_regexp_extract::regexp_extract;
use std::sync::Arc;

async fn extract(
    values: Vec<Option<&str>>,
    pattern: &str,
    idx: Option<i64>,
) -> Result<Vec<Option<String>>> {
    let schema = Arc::new(Schema::new(vec![Field::new("s", DataType::Utf8, true)]));
    let batch = RecordBatch::try_new(
        schema,
        vec![Arc::new(StringArray::from(values))],
    )?;
    let ctx = SessionContext::new();
    let df = ctx.read_batch(batch)?.select(vec![regexp_extract(
        col("s"),
        lit(pattern),
        idx.map(lit),
    )])?;
    strings(df.collect().await?)
}

async fn extract_columns(
    values: Vec<Option<&str>>,
    patterns: Vec<Option<&str>>,
    indexes: Vec<Option<i64>>,
) -> Result<Vec<Option<String>>> {
    let schema = Arc::new(Schema::new(vec![
        Field::new("s", DataType::Utf8, true),
        Field::new("pattern", DataType::Utf8, true),
        Field::new("idx", DataType::Int64, true),
    ]));
    let batch = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(StringArray::from(values)),
            Arc::new(StringArray::from(patterns)),
            Arc::new(Int64Array::from(indexes)),
        ],
    )?;
    let ctx = SessionContext::new();
    let df = ctx.read_batch(batch)?.select(vec![regexp_extract(
        col("s"),
        col("pattern"),
        Some(col("idx")),
    )])?;
    strings(df.collect().await?)
}

fn strings(batches: Vec<RecordBatch>) -> Result<Vec<Option<String>>> {
    let mut out = Vec::new();
    for batch in batches {
        let column = batch.column(0).as_string::<i32>();
        out.extend(column.iter().map(|value| value.map(str::to_string)));
    }
    Ok(out)
}

#[tokio::test]
async fn extracts_requested_group_and_keeps_nulls() -> Result<()> {
    let values = vec![Some("100-200"), Some("foo"), None];

    assert_eq!(
        extract(values.clone(), r"(\d+)-(\d+)", Some(1)).await?,
        vec![Some("100".to_string()), Some(String::new()), None]
    );
    assert_eq!(
        extract(values.clone(), r"(\d+)-(\d+)", Some(2)).await?,
        vec![Some("200".to_string()), Some(String::new()), None]
    );
    assert_eq!(
        extract(values, r"(\d+)-(\d+)", Some(0)).await?,
        vec![Some("100-200".to_string()), Some(String::new()), None]
    );
    Ok(())
}

#[tokio::test]
async fn returns_only_the_first_match() -> Result<()> {
    assert_eq!(
        extract(vec![Some("100-200, 300-400")], r"(\d+)-(\d+)", Some(1)).await?,
        vec![Some("100".to_string())]
    );
    Ok(())
}

#[tokio::test]
async fn unmatched_optional_group_is_empty_string() -> Result<()> {
    assert_eq!(
        extract(vec![Some("aaaac")], r"(a+)(b)?(c)", Some(2)).await?,
        vec![Some(String::new())]
    );
    assert_eq!(
        extract(vec![Some("aaaac")], r"(a+)(b)?(c)", Some(1)).await?,
        vec![Some("aaaa".to_string())]
    );
    Ok(())
}

#[tokio::test]
async fn omitted_index_defaults_to_group_one() -> Result<()> {
    assert_eq!(
        extract(vec![Some("100-200")], r"(\d+)-(\d+)", None).await?,
        vec![Some("100".to_string())]
    );

    let err = extract(vec![Some("1a 2b 14m")], r"\d+", None)
        .await
        .unwrap_err();
    assert!(
        err.to_string().contains("out of range"),
        "default index is 1, and this pattern has no capturing group: {err}"
    );
    assert_eq!(
        extract(vec![Some("1a 2b 14m")], r"\d+", Some(0)).await?,
        vec![Some("1".to_string())]
    );
    Ok(())
}

#[tokio::test]
async fn bad_index_errors_only_when_the_row_matches() -> Result<()> {
    assert_eq!(
        extract(vec![Some("foo")], r"(\d+)", Some(5)).await?,
        vec![Some(String::new())]
    );
    assert_eq!(
        extract(vec![Some("foo")], r"(\d+)", Some(-1)).await?,
        vec![Some(String::new())]
    );

    let too_high = extract(vec![Some("100")], r"(\d+)", Some(5))
        .await
        .unwrap_err();
    assert!(too_high.to_string().contains("out of range"), "{too_high}");

    let negative = extract(vec![Some("100")], r"(\d+)", Some(-1))
        .await
        .unwrap_err();
    assert!(negative.to_string().contains("out of range"), "{negative}");
    Ok(())
}

#[tokio::test]
async fn invalid_pattern_fails_the_call() -> Result<()> {
    let err = extract(vec![Some("abc")], "(?l)", Some(0))
        .await
        .unwrap_err();
    assert!(
        err.to_string().contains("invalid regexp_extract pattern"),
        "{err}"
    );
    Ok(())
}

#[tokio::test]
async fn null_pattern_literal_nulls_every_row() -> Result<()> {
    let schema = Arc::new(Schema::new(vec![Field::new("s", DataType::Utf8, true)]));
    let batch = RecordBatch::try_new(
        schema,
        vec![Arc::new(StringArray::from(vec![Some("100-200"), Some("foo")]))],
    )?;
    let ctx = SessionContext::new();
    let df = ctx.read_batch(batch)?.select(vec![regexp_extract(
        col("s"),
        lit(ScalarValue::Utf8(None)),
        Some(lit(1i64)),
    )])?;

    assert_eq!(
        strings(df.collect().await?)?,
        vec![None, None]
    );
    Ok(())
}

#[tokio::test]
async fn pattern_and_index_may_be_columns() -> Result<()> {
    assert_eq!(
        extract_columns(
            vec![Some("100-200"), Some("aaaac"), Some("foo"), Some("zz")],
            vec![Some(r"(\d+)-(\d+)"), Some(r"(a+)(b)?(c)"), Some(r"(\d+)"), None],
            vec![Some(1), Some(2), Some(0), Some(1)],
        )
        .await?,
        vec![
            Some("100".to_string()),
            Some(String::new()),
            Some(String::new()),
            None,
        ]
    );
    assert_eq!(
        extract_columns(
            vec![Some("100-200")],
            vec![Some(r"(\d+)")],
            vec![None],
        )
        .await?,
        vec![None]
    );
    Ok(())
}
