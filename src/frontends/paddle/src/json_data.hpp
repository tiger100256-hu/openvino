namespace ov {
namespace frontend {
namespace paddle {
namespace json {

    enum Dialet {
        Builtin = 0;
        Operator = 1;
        ControlFlow = 2;
        CustomOp = 3;
        Dist = 4;
    };
    enum TypeType {
        UNDEFINED;
        BF16;
        F16;
        F32;
        F64;
        I8;
        UI8;
        I16;
        I32;
        I64;
        INDEX;
        BOOL;
        C64;
        C128;
        F8E4M3FN;
        F8E5M2;
        DTENSOR;
        VEC;
    };
    struct Port {
       uint64_t id;
       std::string type;
       TypeType precision;
       std::vector<int64_t> shapes;
       std::string layout;
       std::vector<std::vector<size_t>> lod;
       uint64_t offset;
    };
    struct OP {
       std::string name;
       std::string type;
       std::vector<uint64_t> inputIds;
       std::vector<Port> outputPorts;
       bool is_distributed;
       bool is_parameter;
       bool need_clip;
       std::string distAttrs;
       std::string attrs;
       std::string outAttrs;
       std::string quantAttrs;
    };
    struct ConstOP : OP {
    }
    struct Block {
        std::string name;
        std::vector<std::string> args;
        std::vector<OP> ops;
    };
    struct Region{
        std::string id;
        std::vector<Block> blocks;
    };
    Graph {
        std::string magic;
        uint64_t version;
        bool trainable;
        std::vector<Region> regions;
    }
    TypeType convertFromStringToType(std::string type) {
      const static std::map<std::string, TypeType> map = {
          {"t_undefined", UNDEFINED},
          {"t_bf16", BF16},
          {"t_f16", F16},
          {"t_f32", F32},
          {"t_f64", F64},
          {"t_i8", I8},
          {"t_ui8", UI8},
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
      if (iter != map.end) {
         return  iter->second;
      } else {
         return UNDEFINED;
      }
    }
}  // namespace json
}  // namespace paddle
}  // namespace frontend
}  // namespace ov
