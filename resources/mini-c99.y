%epp "LESS" "<"
%epp "GREATER" ">"
%epp "LE" "<="
%epp "GE" ">="
%epp "EQ" "=="
%epp "NEQ" "!="
%epp "ASSIGN" "="
%epp "PLUS" "+"
%epp "INC_OP" "++"
%epp "MINUS" "-"
%epp "DEC_OP" "--"
%epp "MULTIPLY" "*"
%epp "DIVIDE" "/"
%epp "MOD" "%"
%epp "NOT" "!"
%epp "PLUSEQ" "+="
%epp "MINUSEQ" "-="
%epp "MULTEQ" "*="
%epp "DIVEQ" "/="
%epp "MODEQ" "%="
%epp "AMPERSAND" "&"
%epp "BITWISE_OR" "|"
%epp "BITWISE_XOR" "^"
%epp "LOGICAL_AND" "&&"
%epp "LOGICAL_OR" "||"
%epp "QUESTION" "?"
%epp "LEFT_PAREN" "("
%epp "RIGHT_PAREN" ")"
%epp "LEFT_BRACE" "{"
%epp "RIGHT_BRACE" "}"
%epp "LEFT_BRACKET" "["
%epp "RIGHT_BRACKET" "]"
%epp "SEMICOLON" ";"
%epp "COLON" ":"
%epp "COMMA" ","

%%

program : declaration_list;


declaration_list: declaration_or_def declaration_list | declaration_or_def;


declaration_or_def: 
    function_def
    | var_declaration
    | assigns "SEMICOLON"
    ;


function_def:
    type "ID" "LEFT_PAREN" parameter_list "RIGHT_PAREN" statement
    |  "ID" "LEFT_PAREN" parameter_list "RIGHT_PAREN" statement
;

parameter_list:
    // empty
    | type lvalue
    | parameter_list "COMMA" type lvalue
;

type : 
    "INT"
    | "VOID"
    | "FLOAT"
    | "DOUBLE"
    | "CHAR"
    | type "MULTIPLY"
    ;

statement_list:
  statement
| statement_list statement
;

statement: 
    | var_declaration
    | expression "SEMICOLON" 
    | assigns "SEMICOLON" 
    | "IF" "LEFT_PAREN" expression "RIGHT_PAREN" statement
    | "IF" "LEFT_PAREN" expression "RIGHT_PAREN" statement "ELSE" statement
    | "WHILE" "LEFT_PAREN" expression "RIGHT_PAREN" statement
    | for_statement
    | "RETURN" "SEMICOLON"
    | "RETURN" expression "SEMICOLON" 
    | "LEFT_BRACE" "RIGHT_BRACE"
    | "LEFT_BRACE" statement_list "RIGHT_BRACE"
    | "BREAK" "SEMICOLON"
    | "CONTINUE" "SEMICOLON"
;

var_declaration:  
    | type lvalues "SEMICOLON";


lvalues:
    lvalue
    | lvalue "COMMA" lvalues
    ;

lvalue:
    "ID"
    | pointer_access_lval
    | arr
    | "LEFT_PAREN" lvalue "RIGHT_PAREN"
    ;

pointer_access_lval:
    "MULTIPLY" "ID"
    | "MULTIPLY" pointer_access_lval
    ;

arr: "ID" arr_sizes;

arr_sizes: 
    arr_size
    | arr_size arr_sizes;

arr_size: 
    "LEFT_BRACKET" "RIGHT_BRACKET"
    | "LEFT_BRACKET" expression "RIGHT_BRACKET"
    ;

assigns: single_assign | single_assign "COMMA" assigns;

single_assign:
    type lvalues assign_op expression
    | lvalues assign_op expression
    ;

assign_op:
    "ASSIGN"
   | "PLUSEQ"
   | "MINUSEQ"
   | "MULTEQ"
   | "DIVEQ"
   | "MODEQ"
   ;

for_statement:
    "FOR" "LEFT_PAREN" expression_or_assign "SEMICOLON"  expression "SEMICOLON"  expression_or_assign "RIGHT_PAREN" statement
;

expression_or_assign: expression | assigns;

expression:
    "ID"
    | "NUM"
    | "STRING_LITERAL"
    | call
    | "ID"
    | "NOT" expression
    | "MINUS" expression
    | arr
    | "AMPERSAND" expression // address
    | pointer_access_rval
    | "LEFT_PAREN" expression "RIGHT_PAREN"
    | expression "LEFT_BRACKET" expression "RIGHT_BRACKET"
    | expression "PLUS" expression
    | expression "MINUS" expression
    | expression "MULTIPLY" expression
    | expression "DIVIDE" expression
    | expression "MOD" expression
    | expression "LESS" expression
    | expression "GREATER" expression
    | expression "LE" expression
    | expression "GE" expression 
    | expression "EQ" expression
    | expression "NEQ" expression
    | expression "LOGICAL_AND" expression
    | expression "LOGICAL_OR" expression
    | expression "BITWISE_OR" expression
    | expression "BITWISE_XOR" expression
    | expression "AMPERSAND" expression
    | expression "QUESTION" expression "COLON" expression
    | "ID" "INC_OP"
    | "ID" "DEC_OP"
    | "INC_OP" "ID"
    | "DEC_OP" "ID"
;

pointer_access_rval: "MULTIPLY" expression;

call:
    "ID" "LEFT_PAREN" call_arg_list "RIGHT_PAREN"
    ;

call_arg_list:
    //empty
    | expression
    | expression "COMMA" call_arg_list
    ; 
 