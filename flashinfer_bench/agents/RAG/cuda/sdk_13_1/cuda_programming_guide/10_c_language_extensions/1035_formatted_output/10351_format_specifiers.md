# 10.35.1. Format Specifiers

### 10.35.1. Format Specifiers[](#format-specifiers "Permalink to this headline")

As for standard `printf()`, format specifiers take the form: `%[flags][width][.precision][size]type`

The following fields are supported (see widely-available documentation for a complete description of all behaviors):

* Flags: `'#' ' ' '0' '+' '-'`
* Width: `'*' '0-9'`
* Precision: `'0-9'`
* Size: `'h' 'l' 'll'`
* Type: `"%cdiouxXpeEfgGaAs"`

Note that CUDA’s `printf()`will accept any combination of flag, width, precision, size and type, whether or not overall they form a valid format specifier. In other words, “`%hd`” will be accepted and printf will expect a double-precision variable in the corresponding location in the argument list.