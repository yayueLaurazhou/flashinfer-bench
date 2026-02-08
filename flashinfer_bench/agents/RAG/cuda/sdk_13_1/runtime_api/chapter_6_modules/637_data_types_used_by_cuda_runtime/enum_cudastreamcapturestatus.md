# enum cudaStreamCaptureStatus

Possible stream capture statuses returned by cudaStreamIsCapturing

##### Values

**cudaStreamCaptureStatusNone = 0**
Stream is not capturing
**cudaStreamCaptureStatusActive = 1**
Stream is actively capturing
**cudaStreamCaptureStatusInvalidated = 2**
Stream is part of a capture sequence that has been invalidated, but not terminated