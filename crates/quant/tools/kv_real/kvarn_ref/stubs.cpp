// Stubs for backend functions that llama-kvarn.cpp references but the
// quantise / dequantise routines never call.
#include <cstdlib>
extern "C" {
int ggml_backend_dev_type(void *) { abort(); }
void * ggml_backend_dev_backend_reg(void *) { abort(); }
bool ggml_backend_dev_is_meta(void *) { abort(); }
unsigned long ggml_backend_meta_device_count(void *) { abort(); }
void * ggml_backend_meta_device_get(void *, unsigned long) { abort(); }
void * ggml_backend_reg_get_proc_address(void *, const char *) { abort(); }
}
