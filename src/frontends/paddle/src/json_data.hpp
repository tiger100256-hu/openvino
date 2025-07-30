#pragma once
#include "openvino/core/any.hpp"
#include "openvino/core/type/element_type.hpp"
#include <nlohmann/json.hpp>
#include <limits>
#include <type_traits>
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

struct PortDesc {
   TypeType precision;
   std::vector<int64_t> shapes;
   std::string layout;
   std::vector<std::vector<size_t>> lod;
   uint64_t offset = 0;
};

struct Port {
   int64_t id = 0;
   std::string type;
   std::vector<PortDesc> descs;
   TypeType get_precision() const;
   const std::vector<int64_t>& get_shapes() const;
   const std::vector<size_t> get_static_shapes() const;
   const std::string& get_layout() const;
   bool used = false;
};
struct Region;
class OP {
public:
   OP(const nlohmann::json& json_data_):json_data(json_data_){}
   const std::vector<int64_t>& get_sub_inputs_ids(const size_t block_idx) const;
   const std::vector<int64_t>& get_sub_outputs_ids(const size_t block_idx) const;
   std::string name;
   std::string type;
   std::vector<int64_t> inputIds;
   std::set<int64_t> unusedInputIds;
   std::vector<Port> outputPorts;
   bool is_distributed = false;
   bool is_parameter = false;
   bool need_clip = false;
   std::string distAttrs;
   std::string attrs;
   std::string outAttrs;
   std::string quantAttrs;
   std::vector<size_t> sub_block_idxs;
   std::vector<std::shared_ptr<Region>> sub_region_vecs;
   const nlohmann::json& json_data;
};
struct Block {
    std::string name;
    uint64_t id;
    std::vector<Port> args;
    std::vector<OP> ops;
    std::vector<int64_t> input_ids;
    std::vector<int64_t> output_ids;
};
struct Region{
    std::string name;
    std::vector<Block> blocks;
};
struct Graph {
    std::string magic;
    uint64_t version = 0;
    bool trainable = false;
    std::vector<std::shared_ptr<Region>> regions;
    nlohmann::json json_data;
};
TypeType convertFromStringToType(std::string type);
void decodeRegion(const nlohmann::json& json, std::shared_ptr<Region> region);
void decodeBlock(const nlohmann::json& json, Block& block);
void decodeOP(const nlohmann::json& json, OP& op);
void decodeConst(const nlohmann::json& json, OP& op);
void decodeOutPorts(const nlohmann::json& json, OP& op);
void decodePort(const nlohmann::json& json, Port& port);
void decodePortDesc(const nlohmann::json& json, PortDesc& desc);
void decodeArgs(const nlohmann::json& json, Port& port);
template<typename T>
T decode_simple_float_attr_value(const nlohmann::json& json) {
    if (json.contains("VD")) {
        if (json["VD"] == "-INF") {
             return -std::numeric_limits<T>::infinity();
        } else if (json["VD"] == "INF") {
             return std::numeric_limits<T>::infinity();
        } else {
            throw std::runtime_error("unkown VD value");
            return std::numeric_limits<T>::infinity();
        }
    }
    return json.at("D").template get<T>();
};
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
ov::element::Type convert_to_ov_type_from_str(const std::string& type);
}  // namespace json
}  // namespace paddle
}  // namespace frontend
}  // namespace ov
