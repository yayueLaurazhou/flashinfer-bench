# enum cudaDriverEntryPointQueryResult

Enum for status from obtaining driver entry points, used with cudaApiGetDriverEntryPoint

##### Values

**cudaDriverEntryPointSuccess = 0**
Search for symbol found a match
**cudaDriverEntryPointSymbolNotFound = 1**
Search for symbol was not found
**cudaDriverEntryPointVersionNotSufficent = 2**
Search for symbol was found but version wasn't great enough