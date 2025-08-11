#include "json_data.hpp"
#include "openvino/frontend/frontend.hpp"
namespace ov {
namespace frontend {
namespace paddle {
namespace json {
void decodeRegion(const nlohmann::json& json, std::shared_ptr<Region> region){
    region->name = json.at("#").template get<std::string>();
    auto& blocksJson = json.at("blocks");
    for (auto& blockJson : blocksJson) {
        Block newBlock;
        decodeBlock(blockJson, newBlock);
        region->blocks.push_back(std::move(newBlock));
    }
}
void decodeBlock(const nlohmann::json& json, Block& block) {
    block.name = json.at("#").template get<std::string>();
    auto argsJson = json.at("args");
    for (auto& argJson : argsJson) {
        Port newPort;
        decodeArgs(argJson, newPort);
        block.args.push_back(std::move(newPort));
    }
    auto& opsJson = json.at("ops");
    for (auto& opJson : opsJson) {
        OP newOp(opJson);
        decodeOP(opJson, newOp);
        block.ops.push_back(std::move(newOp));
    }
}
void decodeOP(const nlohmann::json& json, OP& op) {
    op.type = json.at("#").template get<std::string>();
    if (op.type == "p") {
        decodeConst(json, op);
    } else {
        size_t pos = op.type.find('.');
        auto dialet = op.type.substr(0, pos);
        if (pos != std::string::npos) {
            op.type = op.type.substr(pos + 1);
            std::cout << "dialet:" << dialet << " type:" << op.type << std::endl;
            // the name sum is conflict with paddle 2.0 op sum
            if (op.type == "sum") {
                op.type = "reduce_sum";
            } else if (op.type == "split" && dialet == "1") {
                op.type = "split_with_num";
            } else if (op.type == "if" || op.type == "while") {
                //decode sub graph
                auto& regionsJson = json.at("regions");
                for (auto& regionJson : regionsJson) {
                    auto sub_region = std::make_shared<Region>();
                    json::decodeRegion(regionJson, sub_region);
                    op.sub_region_vecs.push_back(sub_region);
                }
            }
        }
        auto& inputsJson = json.at("I");
        for (auto& inputJson : inputsJson) {
            auto inputId = inputJson.at("%").template get<uint64_t>();
            op.inputIds.push_back(inputId);
        }
        // op.attrs = json.at("A").dump(); // decode in decode_json.cpp
    }
    // op.outAttrs = json.at("OA").dump();// save it, maybe need in future
    decodeOutPorts(json, op);
}
void decodeConst(const nlohmann::json& json, OP& op) {
    auto& attrJson = json.at("A");
    op.is_distributed = (attrJson.at(0).template get<int32_t>() != 0);
    op.is_parameter = (attrJson.at(1).template get<int32_t>() != 0);
    op.need_clip = (attrJson.at(2).template get<int32_t>() != 0);
    op.name = attrJson.at(3).template get<std::string>();
    op.distAttrs = json.at("DA").dump();// save it, maybe need in future
    op.quantAttrs = json.at("QA").dump();// save it, maybe need in future
}
void decodeOutPorts(const nlohmann::json& json, OP& op) {
    auto& outPortsJson = json.at("O");
    if (outPortsJson.is_array()) {
        for (auto& outPortJson : outPortsJson) {
            Port newPort;
            decodePort(outPortJson, newPort);
            op.outputPorts.push_back(std::move(newPort));
        }
    } else {
        Port newPort;
        decodePort(outPortsJson, newPort);
        op.outputPorts.push_back(std::move(newPort));
    }
    // if (op.type = "while") {
    //    // add a fake bool output
    //    Port fakePort;
    //    auto last_port = op.outputPorts.back()
    //    fakePort.id = std::numeric_limits<std::int64_t>::max() - last_port.id
    //    op.outputPorts.push_back(std::move(fakePort));
    // }
}

void decodePortDesc(const nlohmann::json& json, PortDesc& desc) {
    auto& data = json.at("D");
    auto precisionString = data.at(0).at("#").template get<std::string>();
    size_t pos = precisionString.find('.');
    if (pos != std::string::npos) {
        precisionString = precisionString.substr(pos + 1);
    }
    desc.precision = convertFromStringToType(precisionString);
    desc.shapes = data.at(1).template get<std::vector<int64_t>>();
    desc.layout = data.at(2).template get<std::string>();
    if (data.size() > 3) {
        //desc.lod = data.at(3).template get<std::vector<std::vector<size_t>>>(); // save it, maybe need in future
        desc.offset = data.at(4).template get<size_t>();// save it, maybe need in future
    }
}
void decodePort(const nlohmann::json& json, Port& port) {
    port.id = json.at("%").template get<uint64_t>();
    auto& typeTypeJson = json.at("TT");
    auto port_type = typeTypeJson.at("#").template get<std::string>();
    if (port_type == "NULL") {
        return;
    }
    auto pos = port_type.find(".");
    port.type = port_type.substr(pos + 1);
    if (port.type == "t_vec") {
        auto& data = typeTypeJson.at("D");
        for (auto& portDescJson : data) {
            PortDesc newPortDesc;
            decodePortDesc(portDescJson, newPortDesc);
            port.descs.push_back(std::move(newPortDesc));
        }
    } else {
        PortDesc newPortDesc;
        decodePortDesc(typeTypeJson, newPortDesc);
        port.descs.push_back(std::move(newPortDesc));
    }
}
void decodeArgs(const nlohmann::json& json, Port& port) {
    port.id = json.at("#").template get<uint64_t>();
    auto& typeTypeJson = json.at("TT");
    auto port_type = typeTypeJson.at("#").template get<std::string>();
    if (port_type == "NULL") {
        return;
    }
    auto pos = port_type.find(".");
    port.type = port_type.substr(pos + 1);
    if (port.type == "t_vec") {
        auto& data = typeTypeJson.at("D");
        for (auto& portDescJson : data) {
            PortDesc newPortDesc;
            decodePortDesc(portDescJson, newPortDesc);
            port.descs.push_back(std::move(newPortDesc));
        }
    } else {
        PortDesc newPortDesc;
        decodePortDesc(typeTypeJson, newPortDesc);
        port.descs.push_back(std::move(newPortDesc));
    }
}

TypeType convertFromStringToType(std::string type) {
  const static std::map<std::string, TypeType> map = {
      {"t_undefined", UNDEFINED},
      {"t_bf16", BF16},
      {"t_f16", F16},
      {"t_f32", F32},
      {"t_f64", F64},
      {"t_i8", I8},
      {"t_u8", U8},
      {"t_i16", I16},
      {"t_i32", I32},
      {"t_i64", I64},
      {"t_index", INDEX},
      {"t_bool", BOOL},
      {"t_c64", C64},
      {"t_c128", C128},
      {"t_f8e4m3fn", F8E4M3FN},
      {"t_f8e5m2", F8E5M2},
      {"t_dtensor", DTENSOR},
      {"t_vec", VEC}
  };
  auto iter = map.find(type);
  if (iter != map.end()) {
     return  iter->second;
  } else {
     return UNDEFINED;
  }
}
ov::Any decode_vector_attrs(const nlohmann::json& attrs) {
    auto& attr_data = attrs.at("D");
    for(auto& attr : attr_data) {
        std::string attr_type = attr.at("#").template get<std::string>();
        auto pos = attr_type.find(".");
        attr_type = attr_type.substr(pos + 1);
        if (attr_type  == "a_i32") {
            return ov::Any(decode_vector_attrs_value<int32_t>(attrs));
        } else if (attr_type  == "a_i64") {
            return ov::Any(decode_vector_attrs_value<int64_t>(attrs));
        } else if (attr_type  == "a_bool") {
            return ov::Any(decode_vector_attrs_value<bool>(attrs));
        } else if (attr_type  == "a_str") {
            return ov::Any(decode_vector_attrs_value<std::string>(attrs));
        } else if (attr_type  == "a_f32") {
            return ov::Any(decode_vector_attrs_value<float>(attrs));
        } else if (attr_type  == "a_f64") {
            return ov::Any(decode_vector_attrs_value<double>(attrs));
        } else {
            FRONT_END_GENERAL_CHECK(false, "unsupport vector attr type:", attr_type);
            break;
        }
    }
    return {};
}

ov::Any decode_attr(const nlohmann::json& attr) {
    auto& attr_type = attr.at("AT");
    std::string attr_type_name = attr_type.at("#").template get<std::string>();
    auto pos = attr_type_name.find(".");
    attr_type_name = attr_type_name.substr(pos + 1);
    if (attr_type_name  == "a_i32") {
        return ov::Any(decode_simple_attr_value<int32_t>(attr_type));
    } else if (attr_type_name  == "a_i64") {
        return ov::Any(decode_simple_attr_value<int64_t>(attr_type));
    } else if (attr_type_name  == "a_bool") {
        return ov::Any(decode_simple_attr_value<bool>(attr_type));
    } else if (attr_type_name  == "a_str") {
        return ov::Any(decode_simple_attr_value<std::string>(attr_type));
    } else if (attr_type_name  == "a_f32") {
        return ov::Any(decode_simple_float_attr_value<float>(attr_type));
    } else if (attr_type_name  == "a_f64") {
        return ov::Any(decode_simple_float_attr_value<double>(attr_type));
    } else if (attr_type_name  == "a_dtype") {
        std::string dtype_str = decode_simple_attr_value<std::string>(attr_type);
        return ov::Any(convert_to_ov_type_from_str(dtype_str));
    } else if (attr_type_name  == "a_array") {
        return ov::Any(decode_vector_attrs(attr_type));
    }  else {
        FRONT_END_GENERAL_CHECK(false, "unsupport attr type:", attr_type);
    }
    return {};
}
ov::element::Type convert_to_ov_type(TypeType type) {
    static const std::map<json::TypeType, ov::element::Type> type_map{
        {BOOL, ov::element::boolean},
        {I16, ov::element::i16},
        {I32, ov::element::i32},
        {I64, ov::element::i64},
        {F16, ov::element::f16},
        {F32, ov::element::f32},
        {F64, ov::element::f64},
        {U8, ov::element::u8},
        {I8, ov::element::i8},
        {BF16, ov::element::bf16}};
    auto it = type_map.find(type);
    OPENVINO_ASSERT(it != type_map.end(), "Cannot convert PDPD type to ov::element::Type");
    return it->second;
}
ov::element::Type convert_to_ov_type_from_str(const std::string& type) {
    static const std::map<std::string, ov::element::Type> type_map{
        {"int16", ov::element::i16},
        {"int32", ov::element::i32},
        {"int64", ov::element::i64},
        {"float16", ov::element::f16},
        {"float32", ov::element::f32},
        {"float64", ov::element::f64},
        {"bool", ov::element::boolean}
    };
    auto it = type_map.find(type);
    OPENVINO_ASSERT(it != type_map.end(), "Cannot convert PDPD str ",  type, " to ov::element::Type");
    return it->second;
}

TypeType Port::get_precision() const {
   assert(descs.size() >= 1);
   return descs[0].precision;
}

const std::vector<int64_t>& Port::get_shapes() const {
   assert(descs.size() >= 1);
   return descs[0].shapes;
}

const std::vector<size_t> Port::get_static_shapes() const {
    assert(descs.size() >= 1);
    std::vector<size_t> static_shapes(descs[0].shapes.size());
    std::transform(descs[0].shapes.begin(), descs[0].shapes.end(), static_shapes.begin(), [](int64_t v) {
        assert(v >= 0);
        return static_cast<size_t>(v);
    });
    return static_shapes;
}

const std::string& Port::get_layout() const {
   assert(descs.size() >= 1);
   return descs[0].layout;
}

const std::vector<int64_t>& OP::get_sub_inputs_ids(const size_t block_idx) const {
    for(auto& region : sub_region_vecs) {
        for(auto& block : region->blocks) {
           if (block_idx == block.id) {
               return block.input_ids;
           }
        }
    }
    OPENVINO_ASSERT(false, "Cannot find block_idx: ",  block_idx);
}
const std::vector<int64_t>& OP::get_sub_outputs_ids(const size_t block_idx) const {
    for(auto& region : sub_region_vecs) {
        for(auto& block : region->blocks) {
           if (block_idx == block.id) {
               return block.output_ids;
           }
        }
    }
    OPENVINO_ASSERT(false, "Cannot find block_idx: ",  block_idx);
}

}  // namespace json
}  // namespace paddle
}  // namespace frontend
}  // namespace ov
