#include <ie_core.hpp>

int main() {
	InferenceEngine::Core core;
	InferenceEngine::CNNNetwork model = ie.ReadNetwork("sample.xml");
//! [part4]
	// Example 1
	// Compile and load networks:
	ov::CompiledModel compiled_model0 = core.compile_model(model, "AUTO:CPU,GPU,MYRIAD", {{"MODEL_PRIORITY", "HIGH"}});
	ov::CompiledModel compiled_model1 = core.compile_model(model, "AUTO:CPU,GPU,MYRIAD", {{"MODEL_PRIORITY", "MEDIUM"}});
	ov::CompiledModel compiled_model2 = core.compile_model(model, "AUTO:CPU,GPU,MYRIAD", {{"MODEL_PRIORITY", "LOW"}});
	/************
	  Assume that all the devices (CPU, GPU, and MYRIAD) can support all the networks.
   	  Result: compiled_model0 will use GPU, compiled_model1 will use MYRIAD, compiled_model3 will use CPU.
	 ************/

	// Example 2
	// Compile and load networks:
	ov::CompiledModel compiled_model0 = core.compile_model(model, "AUTO:CPU,GPU,MYRIAD", {"MODEL_PRIORITY", "LOW"}});
	ov::CompiledModel compiled_model1 = core.compile_model(model, "AUTO:CPU,GPU,MYRIAD", {"MODEL_PRIORITY", "MEDIUM"}});
	ov::CompiledModel compiled_model2 = core.compile_model(model, "AUTO:CPU,GPU,MYRIAD", {"MODEL_PRIORITY", "LOW"}});
	/************
	  Assume that all the devices (CPU, GPU, and MYRIAD) can support all the networks.
	  Result: compiled_model0 will use GPU, compiled_model1 will use GPU, compiled_model3 will use MYRIAD.
	 ************/
//! [part4]
	return 0;
}
