#include <ie_core.hpp>

int main() {
//! [part6]
	ov::Core core;

	// read a network in IR, PaddlePaddle, or ONNX format
	std::shared_ptr<ov::Model> model = core.read_model("sample.xml");

	// load a network to AUTO and set log level to debug
	ov::CompiledModel compiled_model = core.compile_model(model, “AUTO”, {{"LOG_LEVEL ", "LOG_DEBUG"}});

	// or set log level with set_config and load network
	core.set_config({"LOG_LEVEL", "LOG_DEBUG"}}, "AUTO");
	ov::CompiledModel compiled_model = core.compile_model(model, “AUTO”);

//! [part6]
    return 0;
}
