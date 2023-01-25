%%

program : declaration_list;


declaration_list: declaration_list declaration | declaration;


declaration: function_declaration;


function_declaration:
    type "ID" "LEFT_PAREN" parameter_list "RIGHT_PAREN" statement
    |  "ID" "LEFT_PAREN" parameter_list "RIGHT_PAREN" statement
;

parameter_list:
    | type "ID"
    | parameter_list "COMMA" type "ID"
;

type : "INT";

statement_list:
  statement
| statement_list statement
;

statement: 
     type "ID" "SEMICOLON" 
    | type "ID" "ASSIGN" expression "SEMICOLON" 
    | "ID" "ASSIGN" expression "SEMICOLON"
    | if_statement
    | "WHILE" "LEFT_PAREN" expression "RIGHT_PAREN" statement
    | for_statement
    | "RETURN" expression "SEMICOLON" 
    | expression "SEMICOLON" 
    | "LEFT_BRACE" "RIGHT_BRACE"
    | "LEFT_BRACE" statement_list "RIGHT_BRACE"
;

for_statement:
    "FOR" "LEFT_PAREN" expression "SEMICOLON"  expression "SEMICOLON"  expression "RIGHT_PAREN" statement
    | "FOR" "LEFT_PAREN" type "ID" "ASSIGN" expression "SEMICOLON"  expression "SEMICOLON"  expression "RIGHT_PAREN" statement

;

if_statement:
    matched |
    unmatched
;

matched:
    "IF" "LEFT_PAREN" expression "RIGHT_PAREN" matched "ELSE" matched 
;

unmatched: 
    "IF" "LEFT_PAREN" expression "RIGHT_PAREN" matched |
    "IF" "LEFT_PAREN" expression "RIGHT_PAREN" unmatched |
    "IF" "LEFT_PAREN" expression "RIGHT_PAREN" matched "ELSE" unmatched
    ;


expression:
    | "ID"
    | "NUM"
    | expression "LEFT_BRACKET" expression "RIGHT_BRACKET"
;
 