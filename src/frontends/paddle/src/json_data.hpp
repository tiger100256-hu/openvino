#pragma once
#include "openvino/core/any.hpp"
#include "openvino/core/type/element_type.hpp"
#include <nlohmann/json.hpp>
namespace ov {
namespace frontend {
namespace paddle {
namespace json {
enum Dialet {
    Builtin = 0,
    Operator = 1,
    ControlFlow = 2,
    CustomOp = 3,
    Dist = 4
};
enum TypeType {
    UNDEFINED,
    BF16,
    F16,
    F32,
    F64,
    I8,
    U8,
    I16,
    I32,
    I64,
    INDEX,
    BOOL,
    C64,
    C128,
    F8E4M3FN,
    F8E5M2,
    DTENSOR,
    VEC
};
struct Port {
   uint64_t id = 0;
   std::string type;
   TypeType precision;
   std::vector<size_t> shapes;
   std::string layout;
   std::vector<std::vector<size_t>> lod;
   uint64_t offset = 0;
   bool used = false;
};
class OP {
public:
   OP(const nlohmann::json& json_data_):json_data(json_data_){}
   std::string name;
   std::string type;
   std::vector<uint64_t> inputIds;
   std::vector<Port> outputPorts;
   bool is_distributed = false;
   bool is_parameter = false;
   bool need_clip = false;
   std::string distAttrs;
   std::string attrs;
   std::string outAttrs;
   std::string quantAttrs;
   const nlohmann::json& json_data;
};
struct Block {
    std::string name;
    std::vector<std::string> args;
    std::vector<OP> ops;
};
struct Region{
    std::string name;
    std::vector<Block> blocks;
};
struct Graph {
    std::string magic;
    uint64_t version = 0;
    bool trainable = false;
    std::vector<Region> regions;
    nlohmann::json json_data;
};
TypeType convertFromStringToType(std::string type);
void decodeRegion(const nlohmann::json& json, Region& region);
void decodeBlock(const nlohmann::json& json, Block& block);
void decodeOP(const nlohmann::json& json, OP& op);
void decodeConst(const nlohmann::json& json, OP& op);
void decodeOutPorts(const nlohmann::json& json, OP& op);
void decodePort(const nlohmann::json& json, Port& port);
template<typename T>
T decode_simple_attr_value(const nlohmann::json& json) {
    return json.at("D").template get<T>();
};

template<typename T>
std::vector<T> decode_vector_attrs_value(const nlohmann::json& attrs) {
    auto& attr_data = attrs.at("D");
    std::vector<T> result;
    for(auto& attr : attr_data) {
        T attr_value = attr.at("D").template get<T>();
        result.push_back(std::move(attr_value));
    }
    return result;
};
ov::Any decode_vector_attrs(const nlohmann::json& attrs);
ov::Any decode_attr(const nlohmann::json& attr);
ov::element::Type convert_to_ov_type(TypeType type);
}  // namespace json
}  // namespace paddle
}  // namespace frontend
}  // namespace ov
