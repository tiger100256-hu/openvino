#include <ie_core.hpp>

int main() {
//! [part3]
	ov::Core core;

	// Read a network in IR, PaddlePaddle, or ONNX format:
	std::shared_ptr<ov::Model> model = core.read_model("sample.xml");

	// Load a network to AUTO with Performance Hints enabled:
	// To use the “throughput” mode:
	ov::CompiledModel compiled_model = core.compile_model(model, "AUTO:CPU,GPU", {{"PERFORMANCE_HINT", "THROUGHPUT"}});
	// or the “latency” mode:
	ov::CompiledModel compiledModel3 = core.compile_model(model, "AUTO:CPU,GPU", {{"PERFORMANCE_HINT", "LATENCY"}});
//! [part3]
	return 0;
}
