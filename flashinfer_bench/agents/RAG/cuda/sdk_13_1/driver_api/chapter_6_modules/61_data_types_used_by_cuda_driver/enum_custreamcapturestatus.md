# enum CUstreamCaptureStatus

Possible stream capture statuses returned by cuStreamIsCapturing

###### Values

**CU_STREAM_CAPTURE_STATUS_NONE = 0**
Stream is not capturing
**CU_STREAM_CAPTURE_STATUS_ACTIVE = 1**
Stream is actively capturing
**CU_STREAM_CAPTURE_STATUS_INVALIDATED = 2**
Stream is part of a capture sequence that has been invalidated, but not terminated