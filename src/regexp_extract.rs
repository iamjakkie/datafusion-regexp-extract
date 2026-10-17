use std::{collections::HashMap, sync::Arc};

use datafusion::arrow::{array::{Array, AsArray, StringBuilder, new_null_array}, datatypes::{DataType, Int64Type}};
use datafusion_expr::{ColumnarValue, Documentation, Expr, ScalarFunctionArgs, ScalarUDF, ScalarUDFImpl, Signature, TypeSignature, Volatility, lit};
use datafusion_common::{Result, ScalarValue, exec_err};
use regex::Regex;


#[derive(Debug, PartialEq, Eq, Hash)]
pub struct RegexpExtractFunc {
    signature: Signature,
}

impl Default for RegexpExtractFunc {
    fn default() -> Self {
        Self::new()
    }
}

impl RegexpExtractFunc {
    pub fn new() -> Self {
        use DataType::*;
        Self {
            signature: Signature::one_of(
                vec![
                    TypeSignature::Exact(vec![Utf8View, Utf8View]),
                    TypeSignature::Exact(vec![Utf8, Utf8]),
                    TypeSignature::Exact(vec![LargeUtf8, LargeUtf8]),
                    TypeSignature::Exact(vec![Utf8View, Utf8View, Int64]),
                    TypeSignature::Exact(vec![Utf8, Utf8, Int64]),
                    TypeSignature::Exact(vec![LargeUtf8, LargeUtf8, Int64]),
                ],
                Volatility::Immutable,
            )
        }
    }
}

impl ScalarUDFImpl for RegexpExtractFunc {
    fn name(&self) -> &str {
        "regexp_extract"
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_type(&self, arg_types: &[DataType]) -> Result<DataType> {
        Ok(arg_types[0].clone())
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {

        let mut cache: HashMap<String, Regex> = HashMap::new();

        let subject = &args.args[0];
        let pattern = &args.args[1];
        let idx = args.args.get(2);

        let mut builder = StringBuilder::with_capacity(args.number_rows, 0);

        let scalar_re = match pattern {
            ColumnarValue::Scalar(scalar) => match scalar.try_as_str() {
                Some(Some(pat)) => Some( match Regex::new(pat) {
                    Ok(re) => re,
                    Err(err) => return exec_err!("invalid regexp_extract pattern '{pat}': {err}"),
                }),
                Some(None) => return Ok(ColumnarValue::Array(new_null_array(
                    args.return_type(),
                    args.number_rows,
                ))),
                None => return exec_err!("regexp_extract pattern must be a string"),
            },
            ColumnarValue::Array(_) => None,
        };

        for i in 0..args.number_rows {
            let text: Option<&str> = match subject {
                ColumnarValue::Array(array) => string_at(array, i),
                ColumnarValue::Scalar(scalar) => scalar.try_as_str().flatten(),
            };
            let pat: Option<&str> = match pattern {
                ColumnarValue::Array(array) => string_at(array, i),
                ColumnarValue::Scalar(scalar) => scalar.try_as_str().flatten(),
            };
            let group: Option<i64> = match idx {
                None => Some(1),
                Some(ColumnarValue::Scalar(ScalarValue::Int64(value))) => *value,
                Some(ColumnarValue::Array(array)) => {
                    let values = array.as_primitive::<Int64Type>();
                    if values.is_null(i) { None } else { Some(values.value(i)) }
                }
                Some(_) => return exec_err!("regexp_extract index must be Int64"),
            };

            let re = match scalar_re.as_ref() {
                Some(re) => re,
                None => {
                    let Some(pat) = pat else {
                        builder.append_null();
                        continue;
                    };
                    if !cache.contains_key(pat) {
                        let compiled = match Regex::new(pat) {
                            Ok(re) => re,
                            Err(err) => return exec_err!("invalid regexp_extract pattern '{pat}': {err}")
                        };
                        cache.insert(pat.to_string(), compiled);
                    }
                    cache.get(pat).unwrap()
                }
            };

            let Some(text) = text else {
                builder.append_null();
                continue;
            };
            let Some(group) = group else {
                builder.append_null();
                continue;
            };
            match re.captures(text) {
                None => builder.append_value(""),
                Some(caps) => {
                    if group < 0 || (group as usize) >= caps.len() {
                        return exec_err!(
                            "regexp_extract group {group} is out of range, pattern has {} groups",
                            caps.len() - 1
                        );
                    }
                    builder.append_value(
                        caps.get(group as usize).map(|m| m.as_str()).unwrap_or(""),
                    );
                }
            }
        }
        Ok(ColumnarValue::Array(Arc::new(builder.finish())))
    }

    fn documentation(&self) -> Option<&Documentation> {
        None
    }
}

pub fn regexp_extract(subject: Expr, pattern: Expr, idx: Option<Expr>) -> Expr {
    let mut args = vec![subject, pattern];
    args.push(idx.unwrap_or_else(|| lit(1i64)));
    ScalarUDF::new_from_impl(RegexpExtractFunc::new()).call(args)
}

fn string_at(array: &dyn Array, i: usize) -> Option<&str> {
    if array.is_null(i) {
        return None;
    }
    match array.data_type() {
        DataType::Utf8 => Some(array.as_string::<i32>().value(i)),
        DataType::LargeUtf8 => Some(array.as_string::<i64>().value(i)),
        DataType::Utf8View => Some(array.as_string_view().value(i)),
        _ => None
    }
}