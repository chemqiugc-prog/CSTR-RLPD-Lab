# Browser policies

| File | Role | Input |
|---|---|---:|
| `rlpd_cstr_upstream.onnx` | Upstream legacy-CSTR RLPD reference | analyzer contract |
| `rlpd_cstr_jacketed_p24_official.onnx` | Retrained standard P2.4 RLPD reference | 8D analyzer contract |
| `rlpd_cstr_jacketed_p24_field_v1.onnx` | Current P2.4 field-observed policy with external V4 | 9D, no concentration |

ONNX contains only the Actor used for inference. It does not contain the
Critics, replay buffer, PID, V4, SIS, reward, or process model.
