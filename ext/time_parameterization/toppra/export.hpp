// Stand-in for the header CMake's generate_export_header() writes when
// toppra is built as its own library.  Here the vendored sources are
// compiled straight into the _time_parameterization module.
#pragma once

#define TOPPRA_EXPORT
#define TOPPRA_NO_EXPORT
#define TOPPRA_DEPRECATED __attribute__((__deprecated__))
