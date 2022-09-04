use super::gambit;
use super::json;
use cfr::Game;
use std::io::Read;

pub fn from_reader(mut reader: impl Read) -> (Game<String, String>, f64) {
    let mut buff = String::new();
    reader.read_to_string(&mut buff).unwrap();
    if let Ok(res) = json::from_str(&buff) {
        res
    } else if let Ok(res) = gambit::from_str(&buff) {
        res
    } else {
        panic!("couldn't parse any known format; try specifying your format with `--input-format` : https://github.com/erikbrinkman/cfr#auto-error");
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_json() {
        super::from_reader(r#"{ "terminal": 0.0 }"#.as_bytes());
    }

    #[test]
    fn test_gambit() {
        super::from_reader(r#"EFG 2 R "" { "" "" } t "" 1 { 0 0 }"#.as_bytes());
    }

    #[test]
    #[should_panic(expected = "couldn't parse any known format")]
    fn test_error() {
        super::from_reader("random".as_bytes());
    }
}
