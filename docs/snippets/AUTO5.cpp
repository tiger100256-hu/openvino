#include <ie_core.hpp>

int main() {
//! [part5]
	ov::Core core;

	// Read a network in IR, PaddlePaddle, or ONNX format:
	std::shared_ptr<ov::Model> model = core.read_model("sample.xml");

	// Configure the VPUX and the Myriad devices separately and load the network to Auto-Device plugin:
	core.set_config(vpux_config, "VPUX");
	core.set_config(vpux_config, "MYRIAD");
	ov::CompiledModel compiled_model = core.compile_model(model);

	// Alternatively, you can combine the individual device settings into one configuration and load the network.
	// The AUTO plugin will parse and apply the settings to the right devices.
	// The 'device_name' of "AUTO:VPUX,MYRIAD" will configure auto-device to use devices.
	ov::CompiledModel compiled_model = core.compile_model(model, device_name, full_config);

//! [part5]
    return 0;
}
