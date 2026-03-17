use hashbrown::HashSet;

pub fn tokenize(code: &str, variables: &HashSet<String>) -> Vec<String> {
    const MULTI_CHAR_OPS: &[&str] = &[
        "==", ">=", "<=", "<<=", ">>=", "!=", "&&", "||", "->", "<<", ">>", "::", "##", "^=", "|=",
        "%=", "&=", "*=", "/=", "+=", "-=", "++", "--",
    ];

    let mut tokens = Vec::new();
    let mut current_token = String::new();
    let mut chars = code.chars().peekable();

    while let Some(&c) = chars.peek() {
        match c {
            c if c.is_whitespace() => {
                process_pending_token(&mut current_token, &variables, &mut tokens);
                // consume whitespace
                chars.next();
            }

            '(' | ')' | '{' | '}' | ',' | ';' | '[' | ']' => {
                process_pending_token(&mut current_token, &variables, &mut tokens);
                tokens.push(c.into());
                chars.next();
            }

            // collapse strings
            '"' => {
                process_pending_token(&mut current_token, &variables, &mut tokens);
                chars.next();

                let mut string_literal = String::from("\"");
                let mut escaped = false;

                while let Some(next_c) = chars.next() {
                    string_literal.push(next_c);
                    if escaped {
                        escaped = false;
                    } else if next_c == '\\' {
                        escaped = true;
                    } else if next_c == '"' {
                        break;
                    }
                }

                // handles unterminated strings by consuming until EOF

                tokens.push(string_literal);
            }

            // potential multi-char ops
            '=' | '>' | '<' | '!' | '&' | '|' | '-' | '+' | ':' | '^' | '#' | '%' | '/' => {
                process_pending_token(&mut current_token, &variables, &mut tokens);

                // this never fails because chars.peek() is Some
                let first_char = chars.next().unwrap();
                if let Some(&next_char) = chars.peek() {
                    // check for two-char ops
                    let potential_op = format!("{}{}", first_char, next_char);
                    if MULTI_CHAR_OPS.contains(&potential_op.as_str()) {
                        tokens.push(potential_op);
                        chars.next();
                        continue;
                    }
                }
                // push single char if no two-char op matched or only one char left
                tokens.push(first_char.into());
            }

            // separate * for ptr types and *= op
            '*' => {
                process_pending_token(&mut current_token, &variables, &mut tokens);

                // this never fails because chars.peek() is Some
                let first_char = chars.next().unwrap();
                if let Some(&next_char) = chars.peek() {
                    // check specifically for *=
                    if next_char == '=' {
                        tokens.push("*=".into());
                        // consume the '='
                        chars.next();
                        continue;
                    }
                }
                // push single '*' if not '*='
                tokens.push(first_char.into());
            }

            _ => {
                current_token.push(c);
                chars.next();
            }
        }
    }

    // process any remaining token after the loop
    if !current_token.is_empty() {
        process_token(&current_token, &variables, &mut tokens);
    }

    tokens
}

/// Treats and marks function calls as variables.
/// Adds recognised function names to the provided HashSet of variables.
pub fn tokenize_incl_funcs(code: &str, variables: &mut HashSet<String>) -> Vec<String> {
    let tokens = tokenize(code, variables);

    tokens
        .into_iter()
        .map(|tok| {
            if tok.starts_with("sub_") {
                let func_token = format!("@@{}@@", tok);
                variables.insert(tok);
                func_token
            } else {
                tok
            }
        })
        .collect()
}

#[inline]
fn process_token(token: &str, variables: &HashSet<String>, tokens: &mut Vec<String>) {
    if variables.contains(token) {
        tokens.push(format!("@@{}@@", token));
    } else {
        tokens.push(token.into());
    }
}

#[inline]
fn process_pending_token(
    current: &mut String,
    variables: &HashSet<String>,
    tokens: &mut Vec<String>,
) {
    if !current.is_empty() {
        process_token(current, variables, tokens);
        current.clear();
    }
}
