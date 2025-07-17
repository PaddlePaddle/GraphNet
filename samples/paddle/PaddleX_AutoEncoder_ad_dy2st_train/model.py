class PirProgram_880737988565058868:

  def __init__(self):

    self.parameter_105 = self.Op("builtin.parameter", 105, input_types=[], output_types=[self.t_dtensor([16], self.t_f32())], attrs={"is_parameter":self.a_array(self.a_bool(True)), "is_distributed":self.a_array(self.a_bool(False)), "need_clip":self.a_array(self.a_bool(True)), "persistable":self.a_array(self.a_bool(True)), "stop_gradient":self.a_array(self.a_bool(False)), "trainable":self.a_array(self.a_bool(True)), "parameter_name":self.a_str("linear_1.b_0"), "__operands_symbols_signature__":self.a_array(), "__results_symbols_signature__":self.a_array(self.a_symbol(self.s_null()))})

    self.parameter_106 = self.Op("builtin.parameter", 106, input_types=[], output_types=[self.t_dtensor([32, 16], self.t_f32())], attrs={"is_parameter":self.a_array(self.a_bool(True)), "is_distributed":self.a_array(self.a_bool(False)), "need_clip":self.a_array(self.a_bool(True)), "persistable":self.a_array(self.a_bool(True)), "stop_gradient":self.a_array(self.a_bool(False)), "trainable":self.a_array(self.a_bool(True)), "parameter_name":self.a_str("linear_1.w_0"), "__operands_symbols_signature__":self.a_array(), "__results_symbols_signature__":self.a_array(self.a_symbol(self.s_null()))})

    self.parameter_107 = self.Op("builtin.parameter", 107, input_types=[], output_types=[self.t_dtensor([32], self.t_f32())], attrs={"is_parameter":self.a_array(self.a_bool(True)), "is_distributed":self.a_array(self.a_bool(False)), "need_clip":self.a_array(self.a_bool(True)), "persistable":self.a_array(self.a_bool(True)), "stop_gradient":self.a_array(self.a_bool(False)), "trainable":self.a_array(self.a_bool(True)), "parameter_name":self.a_str("linear_0.b_0"), "__operands_symbols_signature__":self.a_array(), "__results_symbols_signature__":self.a_array(self.a_symbol(self.s_null()))})

    self.parameter_108 = self.Op("builtin.parameter", 108, input_types=[], output_types=[self.t_dtensor([96, 32], self.t_f32())], attrs={"is_parameter":self.a_array(self.a_bool(True)), "is_distributed":self.a_array(self.a_bool(False)), "need_clip":self.a_array(self.a_bool(True)), "struct_name":self.a_str("/Linear/"), "persistable":self.a_array(self.a_bool(True)), "stop_gradient":self.a_array(self.a_bool(False)), "trainable":self.a_array(self.a_bool(True)), "parameter_name":self.a_str("linear_0.w_0"), "__operands_symbols_signature__":self.a_array(), "__results_symbols_signature__":self.a_array(self.a_symbol(self.s_null()))})

    self.data_109 = self.Op("pd_op.data", 109, input_types=[], output_types=[self.t_dtensor([16, 2, 96], self.t_f32())], attrs={"struct_name":self.a_str("/Linear/"), "stop_gradient":self.a_array(self.a_bool(True)), "name":self.a_str("args_0"), "shape":self.a_intarray(16, 2, 96), "dtype":self.a_dtype("float32"), "place":self.a_place("undefined", 0), "__operands_symbols_signature__":self.a_array(), "__results_symbols_signature__":self.a_array(self.a_symbol(self.s_null()))})

    self.matmul_110 = self.Op("pd_op.matmul", 110, input_types=[self.t_dtensor([16, 2, 96], self.t_f32()), self.t_dtensor([96, 32], self.t_f32())], output_types=[self.t_dtensor([16, 2, 32], self.t_f32())], attrs={"struct_name":self.a_str("/Linear/"), "stop_gradient":self.a_array(self.a_bool(False)), "transpose_x":self.a_bool(False), "transpose_y":self.a_bool(False), "__operands_symbols_signature__":self.a_array(self.a_symbol(self.s_null()), self.a_symbol(self.s_null())), "__results_symbols_signature__":self.a_array(self.a_symbol(self.s_null()))})

    self.add_111 = self.Op("pd_op.add", 111, input_types=[self.t_dtensor([16, 2, 32], self.t_f32()), self.t_dtensor([32], self.t_f32())], output_types=[self.t_dtensor([16, 2, 32], self.t_f32())], attrs={"struct_name":self.a_str("/Linear/"), "stop_gradient":self.a_array(self.a_bool(False)), "__operands_symbols_signature__":self.a_array(self.a_symbol(self.s_null()), self.a_symbol(self.s_null())), "__results_symbols_signature__":self.a_array(self.a_symbol(self.s_null()))})

    self.relu_112 = self.Op("pd_op.relu", 112, input_types=[self.t_dtensor([16, 2, 32], self.t_f32())], output_types=[self.t_dtensor([16, 2, 32], self.t_f32())], attrs={"struct_name":self.a_str("/ReLU/"), "stop_gradient":self.a_array(self.a_bool(False)), "__operands_symbols_signature__":self.a_array(self.a_symbol(self.s_null())), "__results_symbols_signature__":self.a_array(self.a_symbol(self.s_null()))})

    self.full_113 = self.Op("pd_op.full", 113, input_types=[], output_types=[self.t_dtensor([1], self.t_f32())], attrs={"struct_name":self.a_str("/Dropout/"), "stop_gradient":self.a_array(self.a_bool(True)), "shape":self.a_intarray(1), "value":self.a_f64("0.2"), "dtype":self.a_dtype("float32"), "place":self.a_place("cpu"), "__operands_symbols_signature__":self.a_array(), "__results_symbols_signature__":self.a_array(self.a_symbol(self.s_null()))})

    self.dropout_114 = self.Op("pd_op.dropout", 114, input_types=[self.t_dtensor([16, 2, 32], self.t_f32()), self.t_null(), self.t_dtensor([1], self.t_f32())], output_types=[self.t_dtensor([16, 2, 32], self.t_f32()), self.t_dtensor([16, 2, 32], self.t_ui8())], attrs={"struct_name":self.a_str("/Dropout/"), "stop_gradient":self.a_array(self.a_bool(False), self.a_bool(False)), "is_test":self.a_bool(False), "seed":self.a_i32(0), "mode":self.a_str("upscale_in_train"), "fix_seed":self.a_bool(False), "__operands_symbols_signature__":self.a_array(self.a_symbol(self.s_null()), self.a_symbol(self.s_null()), self.a_symbol(self.s_null())), "__results_symbols_signature__":self.a_array(self.a_symbol(self.s_null()), self.a_symbol(self.s_null()))})

    self.matmul_115 = self.Op("pd_op.matmul", 115, input_types=[self.t_dtensor([16, 2, 32], self.t_f32()), self.t_dtensor([32, 16], self.t_f32())], output_types=[self.t_dtensor([16, 2, 16], self.t_f32())], attrs={"struct_name":self.a_str("/Linear_1/"), "stop_gradient":self.a_array(self.a_bool(False)), "transpose_x":self.a_bool(False), "transpose_y":self.a_bool(False), "__operands_symbols_signature__":self.a_array(self.a_symbol(self.s_null()), self.a_symbol(self.s_null())), "__results_symbols_signature__":self.a_array(self.a_symbol(self.s_null()))})

    self.add_116 = self.Op("pd_op.add", 116, input_types=[self.t_dtensor([16, 2, 16], self.t_f32()), self.t_dtensor([16], self.t_f32())], output_types=[self.t_dtensor([16, 2, 16], self.t_f32())], attrs={"struct_name":self.a_str("/Linear_1/"), "stop_gradient":self.a_array(self.a_bool(False)), "__operands_symbols_signature__":self.a_array(self.a_symbol(self.s_null()), self.a_symbol(self.s_null())), "__results_symbols_signature__":self.a_array(self.a_symbol(self.s_null()))})

    self.shadow_output_117 = self.Op("builtin.shadow_output", 117, input_types=[self.t_dtensor([16, 2, 32], self.t_f32())], output_types=[], attrs={"output_name":self.a_str("middle_0"), "__operands_symbols_signature__":self.a_array(self.a_symbol(self.s_null())), "__results_symbols_signature__":self.a_array()})

    self.shadow_output_118 = self.Op("builtin.shadow_output", 118, input_types=[self.t_dtensor([16, 2, 32], self.t_f32())], output_types=[], attrs={"output_name":self.a_str("middle_1"), "__operands_symbols_signature__":self.a_array(self.a_symbol(self.s_null())), "__results_symbols_signature__":self.a_array()})

    self.shadow_output_119 = self.Op("builtin.shadow_output", 119, input_types=[self.t_dtensor([1], self.t_f32())], output_types=[], attrs={"output_name":self.a_str("middle_2"), "__operands_symbols_signature__":self.a_array(self.a_symbol(self.s_null())), "__results_symbols_signature__":self.a_array()})

    self.shadow_output_120 = self.Op("builtin.shadow_output", 120, input_types=[self.t_dtensor([16, 2, 32], self.t_f32())], output_types=[], attrs={"output_name":self.a_str("middle_3"), "__operands_symbols_signature__":self.a_array(self.a_symbol(self.s_null())), "__results_symbols_signature__":self.a_array()})

    self.shadow_output_121 = self.Op("builtin.shadow_output", 121, input_types=[self.t_dtensor([16, 2, 32], self.t_ui8())], output_types=[], attrs={"output_name":self.a_str("middle_4"), "__operands_symbols_signature__":self.a_array(self.a_symbol(self.s_null())), "__results_symbols_signature__":self.a_array()})

    self.shadow_output_122 = self.Op("builtin.shadow_output", 122, input_types=[self.t_dtensor([16, 2, 16], self.t_f32())], output_types=[], attrs={"output_name":self.a_str("middle_5"), "__operands_symbols_signature__":self.a_array(self.a_symbol(self.s_null())), "__results_symbols_signature__":self.a_array()})

    self.shadow_output_123 = self.Op("builtin.shadow_output", 123, input_types=[self.t_dtensor([16, 2, 16], self.t_f32())], output_types=[], attrs={"output_name":self.a_str("output_0"), "__operands_symbols_signature__":self.a_array(self.a_symbol(self.s_null())), "__results_symbols_signature__":self.a_array()})

    self.module_103 = self.Op("builtin.module", 103, input_types=[], output_types=[], attrs={"program":self.a_pointer("0x584e0a20"), "__operands_symbols_signature__":self.a_array(), "__results_symbols_signature__":self.a_array()}, block_positional_arg_names=[[[]]], block_keyword_arg_names=[[{}]], block_positional_arg_types=[[[]]], block_keyword_arg_types=[[[]]], )

    

  def module_103_block00(self, call):

    def ret_lambda_module_103_block00():

      parameter_1050, = call(self.parameter_105)

      parameter_1060, = call(self.parameter_106)

      parameter_1070, = call(self.parameter_107)

      parameter_1080, = call(self.parameter_108)

      data_1090, = call(self.data_109)

      matmul_1100, = call(self.matmul_110, data_1090, parameter_1080)

      add_1110, = call(self.add_111, matmul_1100, parameter_1070)

      relu_1120, = call(self.relu_112, add_1110)

      full_1130, = call(self.full_113)

      dropout_1140, dropout_1141, = call(self.dropout_114, relu_1120, None, full_1130)

      matmul_1150, = call(self.matmul_115, dropout_1140, parameter_1060)

      add_1160, = call(self.add_116, matmul_1150, parameter_1050)

      call(self.shadow_output_117, matmul_1100)

      call(self.shadow_output_118, relu_1120)

      call(self.shadow_output_119, full_1130)

      call(self.shadow_output_120, dropout_1140)

      call(self.shadow_output_121, dropout_1141)

      call(self.shadow_output_122, matmul_1150)

      call(self.shadow_output_123, add_1160)

    return ret_lambda_module_103_block00

    

  def __call__(self, call, *args, **kwargs):

    self.SetArgs(args)

    self.SetKeywordArgs(kwargs)

    return call(self.module_103, blocks=[[(self.module_103_block00,)]])


class PirProgram_5237713637574795923:

  def __init__(self):

    self.parameter_259 = self.Op("builtin.parameter", 259, input_types=[], output_types=[self.t_dtensor([96], self.t_f32())], attrs={"is_parameter":self.a_array(self.a_bool(True)), "is_distributed":self.a_array(self.a_bool(False)), "need_clip":self.a_array(self.a_bool(True)), "persistable":self.a_array(self.a_bool(True)), "stop_gradient":self.a_array(self.a_bool(False)), "trainable":self.a_array(self.a_bool(True)), "parameter_name":self.a_str("linear_3.b_0"), "__operands_symbols_signature__":self.a_array(), "__results_symbols_signature__":self.a_array(self.a_symbol(self.s_null()))})

    self.parameter_260 = self.Op("builtin.parameter", 260, input_types=[], output_types=[self.t_dtensor([32, 96], self.t_f32())], attrs={"is_parameter":self.a_array(self.a_bool(True)), "is_distributed":self.a_array(self.a_bool(False)), "need_clip":self.a_array(self.a_bool(True)), "persistable":self.a_array(self.a_bool(True)), "stop_gradient":self.a_array(self.a_bool(False)), "trainable":self.a_array(self.a_bool(True)), "parameter_name":self.a_str("linear_3.w_0"), "__operands_symbols_signature__":self.a_array(), "__results_symbols_signature__":self.a_array(self.a_symbol(self.s_null()))})

    self.parameter_261 = self.Op("builtin.parameter", 261, input_types=[], output_types=[self.t_dtensor([32], self.t_f32())], attrs={"is_parameter":self.a_array(self.a_bool(True)), "is_distributed":self.a_array(self.a_bool(False)), "need_clip":self.a_array(self.a_bool(True)), "persistable":self.a_array(self.a_bool(True)), "stop_gradient":self.a_array(self.a_bool(False)), "trainable":self.a_array(self.a_bool(True)), "parameter_name":self.a_str("linear_2.b_0"), "__operands_symbols_signature__":self.a_array(), "__results_symbols_signature__":self.a_array(self.a_symbol(self.s_null()))})

    self.parameter_262 = self.Op("builtin.parameter", 262, input_types=[], output_types=[self.t_dtensor([16, 32], self.t_f32())], attrs={"is_parameter":self.a_array(self.a_bool(True)), "is_distributed":self.a_array(self.a_bool(False)), "need_clip":self.a_array(self.a_bool(True)), "struct_name":self.a_str("/Linear_2/"), "persistable":self.a_array(self.a_bool(True)), "stop_gradient":self.a_array(self.a_bool(False)), "trainable":self.a_array(self.a_bool(True)), "parameter_name":self.a_str("linear_2.w_0"), "__operands_symbols_signature__":self.a_array(), "__results_symbols_signature__":self.a_array(self.a_symbol(self.s_null()))})

    self.data_263 = self.Op("pd_op.data", 263, input_types=[], output_types=[self.t_dtensor([16, 2, -1], self.t_f32())], attrs={"struct_name":self.a_str("/Linear_2/"), "stop_gradient":self.a_array(self.a_bool(False)), "name":self.a_str("args_0"), "shape":self.a_intarray(16, 2, -1), "dtype":self.a_dtype("float32"), "place":self.a_place("undefined", 0), "__operands_symbols_signature__":self.a_array(), "__results_symbols_signature__":self.a_array(self.a_symbol(self.s_null()))})

    self.matmul_264 = self.Op("pd_op.matmul", 264, input_types=[self.t_dtensor([16, 2, -1], self.t_f32()), self.t_dtensor([16, 32], self.t_f32())], output_types=[self.t_dtensor([16, 2, 32], self.t_f32())], attrs={"struct_name":self.a_str("/Linear_2/"), "stop_gradient":self.a_array(self.a_bool(False)), "transpose_x":self.a_bool(False), "transpose_y":self.a_bool(False), "__operands_symbols_signature__":self.a_array(self.a_symbol(self.s_null()), self.a_symbol(self.s_null())), "__results_symbols_signature__":self.a_array(self.a_symbol(self.s_null()))})

    self.add_265 = self.Op("pd_op.add", 265, input_types=[self.t_dtensor([16, 2, 32], self.t_f32()), self.t_dtensor([32], self.t_f32())], output_types=[self.t_dtensor([16, 2, 32], self.t_f32())], attrs={"struct_name":self.a_str("/Linear_2/"), "stop_gradient":self.a_array(self.a_bool(False)), "__operands_symbols_signature__":self.a_array(self.a_symbol(self.s_null()), self.a_symbol(self.s_null())), "__results_symbols_signature__":self.a_array(self.a_symbol(self.s_null()))})

    self.relu_266 = self.Op("pd_op.relu", 266, input_types=[self.t_dtensor([16, 2, 32], self.t_f32())], output_types=[self.t_dtensor([16, 2, 32], self.t_f32())], attrs={"struct_name":self.a_str("/ReLU_1/"), "stop_gradient":self.a_array(self.a_bool(False)), "__operands_symbols_signature__":self.a_array(self.a_symbol(self.s_null())), "__results_symbols_signature__":self.a_array(self.a_symbol(self.s_null()))})

    self.full_267 = self.Op("pd_op.full", 267, input_types=[], output_types=[self.t_dtensor([1], self.t_f32())], attrs={"struct_name":self.a_str("/Dropout_1/"), "stop_gradient":self.a_array(self.a_bool(True)), "shape":self.a_intarray(1), "value":self.a_f64("0.2"), "dtype":self.a_dtype("float32"), "place":self.a_place("cpu"), "__operands_symbols_signature__":self.a_array(), "__results_symbols_signature__":self.a_array(self.a_symbol(self.s_null()))})

    self.dropout_268 = self.Op("pd_op.dropout", 268, input_types=[self.t_dtensor([16, 2, 32], self.t_f32()), self.t_null(), self.t_dtensor([1], self.t_f32())], output_types=[self.t_dtensor([16, 2, 32], self.t_f32()), self.t_dtensor([16, 2, 32], self.t_ui8())], attrs={"struct_name":self.a_str("/Dropout_1/"), "stop_gradient":self.a_array(self.a_bool(False), self.a_bool(False)), "is_test":self.a_bool(False), "seed":self.a_i32(0), "mode":self.a_str("upscale_in_train"), "fix_seed":self.a_bool(False), "__operands_symbols_signature__":self.a_array(self.a_symbol(self.s_null()), self.a_symbol(self.s_null()), self.a_symbol(self.s_null())), "__results_symbols_signature__":self.a_array(self.a_symbol(self.s_null()), self.a_symbol(self.s_null()))})

    self.matmul_269 = self.Op("pd_op.matmul", 269, input_types=[self.t_dtensor([16, 2, 32], self.t_f32()), self.t_dtensor([32, 96], self.t_f32())], output_types=[self.t_dtensor([16, 2, 96], self.t_f32())], attrs={"struct_name":self.a_str("/Linear_3/"), "stop_gradient":self.a_array(self.a_bool(False)), "transpose_x":self.a_bool(False), "transpose_y":self.a_bool(False), "__operands_symbols_signature__":self.a_array(self.a_symbol(self.s_null()), self.a_symbol(self.s_null())), "__results_symbols_signature__":self.a_array(self.a_symbol(self.s_null()))})

    self.add_270 = self.Op("pd_op.add", 270, input_types=[self.t_dtensor([16, 2, 96], self.t_f32()), self.t_dtensor([96], self.t_f32())], output_types=[self.t_dtensor([16, 2, 96], self.t_f32())], attrs={"struct_name":self.a_str("/Linear_3/"), "stop_gradient":self.a_array(self.a_bool(False)), "__operands_symbols_signature__":self.a_array(self.a_symbol(self.s_null()), self.a_symbol(self.s_null())), "__results_symbols_signature__":self.a_array(self.a_symbol(self.s_null()))})

    self.shadow_output_271 = self.Op("builtin.shadow_output", 271, input_types=[self.t_dtensor([16, 2, 32], self.t_f32())], output_types=[], attrs={"output_name":self.a_str("middle_0"), "__operands_symbols_signature__":self.a_array(self.a_symbol(self.s_null())), "__results_symbols_signature__":self.a_array()})

    self.shadow_output_272 = self.Op("builtin.shadow_output", 272, input_types=[self.t_dtensor([16, 2, 32], self.t_f32())], output_types=[], attrs={"output_name":self.a_str("middle_1"), "__operands_symbols_signature__":self.a_array(self.a_symbol(self.s_null())), "__results_symbols_signature__":self.a_array()})

    self.shadow_output_273 = self.Op("builtin.shadow_output", 273, input_types=[self.t_dtensor([1], self.t_f32())], output_types=[], attrs={"output_name":self.a_str("middle_2"), "__operands_symbols_signature__":self.a_array(self.a_symbol(self.s_null())), "__results_symbols_signature__":self.a_array()})

    self.shadow_output_274 = self.Op("builtin.shadow_output", 274, input_types=[self.t_dtensor([16, 2, 32], self.t_f32())], output_types=[], attrs={"output_name":self.a_str("middle_3"), "__operands_symbols_signature__":self.a_array(self.a_symbol(self.s_null())), "__results_symbols_signature__":self.a_array()})

    self.shadow_output_275 = self.Op("builtin.shadow_output", 275, input_types=[self.t_dtensor([16, 2, 32], self.t_ui8())], output_types=[], attrs={"output_name":self.a_str("middle_4"), "__operands_symbols_signature__":self.a_array(self.a_symbol(self.s_null())), "__results_symbols_signature__":self.a_array()})

    self.shadow_output_276 = self.Op("builtin.shadow_output", 276, input_types=[self.t_dtensor([16, 2, 96], self.t_f32())], output_types=[], attrs={"output_name":self.a_str("middle_5"), "__operands_symbols_signature__":self.a_array(self.a_symbol(self.s_null())), "__results_symbols_signature__":self.a_array()})

    self.shadow_output_277 = self.Op("builtin.shadow_output", 277, input_types=[self.t_dtensor([16, 2, 96], self.t_f32())], output_types=[], attrs={"output_name":self.a_str("output_0"), "__operands_symbols_signature__":self.a_array(self.a_symbol(self.s_null())), "__results_symbols_signature__":self.a_array()})

    self.module_257 = self.Op("builtin.module", 257, input_types=[], output_types=[], attrs={"program":self.a_pointer("0x6b30d5a0"), "__operands_symbols_signature__":self.a_array(), "__results_symbols_signature__":self.a_array()}, block_positional_arg_names=[[[]]], block_keyword_arg_names=[[{}]], block_positional_arg_types=[[[]]], block_keyword_arg_types=[[[]]], )

    

  def module_257_block00(self, call):

    def ret_lambda_module_257_block00():

      parameter_2590, = call(self.parameter_259)

      parameter_2600, = call(self.parameter_260)

      parameter_2610, = call(self.parameter_261)

      parameter_2620, = call(self.parameter_262)

      data_2630, = call(self.data_263)

      matmul_2640, = call(self.matmul_264, data_2630, parameter_2620)

      add_2650, = call(self.add_265, matmul_2640, parameter_2610)

      relu_2660, = call(self.relu_266, add_2650)

      full_2670, = call(self.full_267)

      dropout_2680, dropout_2681, = call(self.dropout_268, relu_2660, None, full_2670)

      matmul_2690, = call(self.matmul_269, dropout_2680, parameter_2600)

      add_2700, = call(self.add_270, matmul_2690, parameter_2590)

      call(self.shadow_output_271, matmul_2640)

      call(self.shadow_output_272, relu_2660)

      call(self.shadow_output_273, full_2670)

      call(self.shadow_output_274, dropout_2680)

      call(self.shadow_output_275, dropout_2681)

      call(self.shadow_output_276, matmul_2690)

      call(self.shadow_output_277, add_2700)

    return ret_lambda_module_257_block00

    

  def __call__(self, call, *args, **kwargs):

    self.SetArgs(args)

    self.SetKeywordArgs(kwargs)

    return call(self.module_257, blocks=[[(self.module_257_block00,)]])


class PirProgram_1242071021843173094:

  def __init__(self):

    self.add_grad_278 = self.Op("pd_op.add_grad", 278, input_types=[self.t_dtensor([16, 2, 96], self.t_f32()), self.t_dtensor([96], self.t_f32()), self.t_dtensor([16, 2, 96], self.t_f32())], output_types=[self.t_dtensor([16, 2, 96], self.t_f32()), self.t_dtensor([96], self.t_f32())], attrs={"stop_gradient":self.a_array(self.a_bool(False), self.a_bool(False)), "axis":self.a_i32(-1), "__operands_symbols_signature__":self.a_array(self.a_symbol(self.s_null()), self.a_symbol(self.s_null()), self.a_symbol(self.s_null())), "__results_symbols_signature__":self.a_array(self.a_symbol(self.s_null()), self.a_symbol(self.s_null()))})

    self.matmul_grad_279 = self.Op("pd_op.matmul_grad", 279, input_types=[self.t_dtensor([16, 2, 32], self.t_f32()), self.t_dtensor([32, 96], self.t_f32()), self.t_dtensor([16, 2, 96], self.t_f32())], output_types=[self.t_dtensor([16, 2, 32], self.t_f32()), self.t_dtensor([32, 96], self.t_f32())], attrs={"stop_gradient":self.a_array(self.a_bool(False), self.a_bool(False)), "transpose_x":self.a_bool(False), "transpose_y":self.a_bool(False), "__operands_symbols_signature__":self.a_array(self.a_symbol(self.s_null()), self.a_symbol(self.s_null()), self.a_symbol(self.s_null())), "__results_symbols_signature__":self.a_array(self.a_symbol(self.s_null()), self.a_symbol(self.s_null()))})

    self.dropout_grad_280 = self.Op("pd_op.dropout_grad", 280, input_types=[self.t_dtensor([16, 2, 32], self.t_ui8()), self.t_dtensor([16, 2, 32], self.t_f32()), self.t_dtensor([1], self.t_f32())], output_types=[self.t_dtensor([16, 2, 32], self.t_f32())], attrs={"stop_gradient":self.a_array(self.a_bool(False)), "is_test":self.a_bool(False), "mode":self.a_str("upscale_in_train"), "__operands_symbols_signature__":self.a_array(self.a_symbol(self.s_null()), self.a_symbol(self.s_null()), self.a_symbol(self.s_null())), "__results_symbols_signature__":self.a_array(self.a_symbol(self.s_null()))})

    self.relu_grad_281 = self.Op("pd_op.relu_grad", 281, input_types=[self.t_dtensor([16, 2, 32], self.t_f32()), self.t_dtensor([16, 2, 32], self.t_f32())], output_types=[self.t_dtensor([16, 2, 32], self.t_f32())], attrs={"stop_gradient":self.a_array(self.a_bool(False)), "__operands_symbols_signature__":self.a_array(self.a_symbol(self.s_null()), self.a_symbol(self.s_null())), "__results_symbols_signature__":self.a_array(self.a_symbol(self.s_null()))})

    self.add_grad_282 = self.Op("pd_op.add_grad", 282, input_types=[self.t_dtensor([16, 2, 32], self.t_f32()), self.t_dtensor([32], self.t_f32()), self.t_dtensor([16, 2, 32], self.t_f32())], output_types=[self.t_dtensor([16, 2, 32], self.t_f32()), self.t_dtensor([32], self.t_f32())], attrs={"stop_gradient":self.a_array(self.a_bool(False), self.a_bool(False)), "axis":self.a_i32(-1), "__operands_symbols_signature__":self.a_array(self.a_symbol(self.s_null()), self.a_symbol(self.s_null()), self.a_symbol(self.s_null())), "__results_symbols_signature__":self.a_array(self.a_symbol(self.s_null()), self.a_symbol(self.s_null()))})

    self.matmul_grad_283 = self.Op("pd_op.matmul_grad", 283, input_types=[self.t_dtensor([16, 2, -1], self.t_f32()), self.t_dtensor([16, 32], self.t_f32()), self.t_dtensor([16, 2, 32], self.t_f32())], output_types=[self.t_dtensor([16, 2, -1], self.t_f32()), self.t_dtensor([16, 32], self.t_f32())], attrs={"stop_gradient":self.a_array(self.a_bool(False), self.a_bool(False)), "transpose_x":self.a_bool(False), "transpose_y":self.a_bool(False), "__operands_symbols_signature__":self.a_array(self.a_symbol(self.s_null()), self.a_symbol(self.s_null()), self.a_symbol(self.s_null())), "__results_symbols_signature__":self.a_array(self.a_symbol(self.s_null()), self.a_symbol(self.s_null()))})

    self.shadow_output_284 = self.Op("builtin.shadow_output", 284, input_types=[self.t_dtensor([16, 2, -1], self.t_f32())], output_types=[], attrs={"output_name":self.a_str("input_grad_0"), "__operands_symbols_signature__":self.a_array(self.a_symbol(self.s_null())), "__results_symbols_signature__":self.a_array()})

    self.shadow_output_285 = self.Op("builtin.shadow_output", 285, input_types=[self.t_dtensor([32], self.t_f32())], output_types=[], attrs={"output_name":self.a_str("param_grad_0"), "__operands_symbols_signature__":self.a_array(self.a_symbol(self.s_null())), "__results_symbols_signature__":self.a_array()})

    self.shadow_output_286 = self.Op("builtin.shadow_output", 286, input_types=[self.t_dtensor([16, 32], self.t_f32())], output_types=[], attrs={"output_name":self.a_str("param_grad_1"), "__operands_symbols_signature__":self.a_array(self.a_symbol(self.s_null())), "__results_symbols_signature__":self.a_array()})

    self.shadow_output_287 = self.Op("builtin.shadow_output", 287, input_types=[self.t_dtensor([96], self.t_f32())], output_types=[], attrs={"output_name":self.a_str("param_grad_2"), "__operands_symbols_signature__":self.a_array(self.a_symbol(self.s_null())), "__results_symbols_signature__":self.a_array()})

    self.shadow_output_288 = self.Op("builtin.shadow_output", 288, input_types=[self.t_dtensor([32, 96], self.t_f32())], output_types=[], attrs={"output_name":self.a_str("param_grad_3"), "__operands_symbols_signature__":self.a_array(self.a_symbol(self.s_null())), "__results_symbols_signature__":self.a_array()})

    self.module_258 = self.Op("builtin.module", 258, input_types=[], output_types=[], attrs={"program":self.a_pointer("0x6b30eee0"), "__operands_symbols_signature__":self.a_array(), "__results_symbols_signature__":self.a_array()}, block_positional_arg_names=[[[]]], block_keyword_arg_names=[[{"middle_5": "arg_1798365008", "middle_4": "arg_1798348928", "output_grad_0": "arg_1798365584", "linear_3.w_0": "arg_1481522992", "middle_2": "arg_1798347072", "linear_3.b_0": "arg_1798306704", "middle_3": "arg_1798348720", "middle_1": "arg_1798334304", "linear_2.w_0": "arg_1798084816", "middle_0": "arg_1798334032", "linear_2.b_0": "arg_1798322512", "args_0": "arg_1798315712"}]], block_positional_arg_types=[[[]]], block_keyword_arg_types=[[[self.t_dtensor([16, 2, 96], self.t_f32()), self.t_dtensor([16, 2, 32], self.t_ui8()), self.t_dtensor([16, 2, 96], self.t_f32()), self.t_dtensor([32, 96], self.t_f32()), self.t_dtensor([1], self.t_f32()), self.t_dtensor([96], self.t_f32()), self.t_dtensor([16, 2, 32], self.t_f32()), self.t_dtensor([16, 2, 32], self.t_f32()), self.t_dtensor([16, 32], self.t_f32()), self.t_dtensor([16, 2, 32], self.t_f32()), self.t_dtensor([32], self.t_f32()), self.t_dtensor([16, 2, -1], self.t_f32())]]], )

    

  def module_258_block00(self, call):

    def ret_lambda_module_258_block00(arg_1798365008, arg_1798348928, arg_1798365584, arg_1481522992, arg_1798347072, arg_1798306704, arg_1798348720, arg_1798334304, arg_1798084816, arg_1798334032, arg_1798322512, arg_1798315712):

      add_grad_2780, add_grad_2781, = call(self.add_grad_278, arg_1798365008, arg_1798306704, arg_1798365584)

      matmul_grad_2790, matmul_grad_2791, = call(self.matmul_grad_279, arg_1798348720, arg_1481522992, add_grad_2780)

      dropout_grad_2800, = call(self.dropout_grad_280, arg_1798348928, matmul_grad_2790, arg_1798347072)

      relu_grad_2810, = call(self.relu_grad_281, arg_1798334304, dropout_grad_2800)

      add_grad_2820, add_grad_2821, = call(self.add_grad_282, arg_1798334032, arg_1798322512, relu_grad_2810)

      matmul_grad_2830, matmul_grad_2831, = call(self.matmul_grad_283, arg_1798315712, arg_1798084816, add_grad_2820)

      call(self.shadow_output_284, matmul_grad_2830)

      call(self.shadow_output_285, add_grad_2821)

      call(self.shadow_output_286, matmul_grad_2831)

      call(self.shadow_output_287, add_grad_2781)

      call(self.shadow_output_288, matmul_grad_2791)

    return ret_lambda_module_258_block00

    

  def __call__(self, call, *args, **kwargs):

    self.SetArgs(args)

    self.SetKeywordArgs(kwargs)

    return call(self.module_258, blocks=[[(self.module_258_block00,)]])


class PirProgram_692320333968606381:

  def __init__(self):

    self.add_grad_124 = self.Op("pd_op.add_grad", 124, input_types=[self.t_dtensor([16, 2, 16], self.t_f32()), self.t_dtensor([16], self.t_f32()), self.t_dtensor([16, 2, 16], self.t_f32())], output_types=[self.t_dtensor([16, 2, 16], self.t_f32()), self.t_dtensor([16], self.t_f32())], attrs={"stop_gradient":self.a_array(self.a_bool(False), self.a_bool(False)), "axis":self.a_i32(-1), "__operands_symbols_signature__":self.a_array(self.a_symbol(self.s_null()), self.a_symbol(self.s_null()), self.a_symbol(self.s_null())), "__results_symbols_signature__":self.a_array(self.a_symbol(self.s_null()), self.a_symbol(self.s_null()))})

    self.matmul_grad_125 = self.Op("pd_op.matmul_grad", 125, input_types=[self.t_dtensor([16, 2, 32], self.t_f32()), self.t_dtensor([32, 16], self.t_f32()), self.t_dtensor([16, 2, 16], self.t_f32())], output_types=[self.t_dtensor([16, 2, 32], self.t_f32()), self.t_dtensor([32, 16], self.t_f32())], attrs={"stop_gradient":self.a_array(self.a_bool(False), self.a_bool(False)), "transpose_x":self.a_bool(False), "transpose_y":self.a_bool(False), "__operands_symbols_signature__":self.a_array(self.a_symbol(self.s_null()), self.a_symbol(self.s_null()), self.a_symbol(self.s_null())), "__results_symbols_signature__":self.a_array(self.a_symbol(self.s_null()), self.a_symbol(self.s_null()))})

    self.dropout_grad_126 = self.Op("pd_op.dropout_grad", 126, input_types=[self.t_dtensor([16, 2, 32], self.t_ui8()), self.t_dtensor([16, 2, 32], self.t_f32()), self.t_dtensor([1], self.t_f32())], output_types=[self.t_dtensor([16, 2, 32], self.t_f32())], attrs={"stop_gradient":self.a_array(self.a_bool(False)), "is_test":self.a_bool(False), "mode":self.a_str("upscale_in_train"), "__operands_symbols_signature__":self.a_array(self.a_symbol(self.s_null()), self.a_symbol(self.s_null()), self.a_symbol(self.s_null())), "__results_symbols_signature__":self.a_array(self.a_symbol(self.s_null()))})

    self.relu_grad_127 = self.Op("pd_op.relu_grad", 127, input_types=[self.t_dtensor([16, 2, 32], self.t_f32()), self.t_dtensor([16, 2, 32], self.t_f32())], output_types=[self.t_dtensor([16, 2, 32], self.t_f32())], attrs={"stop_gradient":self.a_array(self.a_bool(False)), "__operands_symbols_signature__":self.a_array(self.a_symbol(self.s_null()), self.a_symbol(self.s_null())), "__results_symbols_signature__":self.a_array(self.a_symbol(self.s_null()))})

    self.add_grad_128 = self.Op("pd_op.add_grad", 128, input_types=[self.t_dtensor([16, 2, 32], self.t_f32()), self.t_dtensor([32], self.t_f32()), self.t_dtensor([16, 2, 32], self.t_f32())], output_types=[self.t_dtensor([16, 2, 32], self.t_f32()), self.t_dtensor([32], self.t_f32())], attrs={"stop_gradient":self.a_array(self.a_bool(False), self.a_bool(False)), "axis":self.a_i32(-1), "__operands_symbols_signature__":self.a_array(self.a_symbol(self.s_null()), self.a_symbol(self.s_null()), self.a_symbol(self.s_null())), "__results_symbols_signature__":self.a_array(self.a_symbol(self.s_null()), self.a_symbol(self.s_null()))})

    self.matmul_grad_129 = self.Op("pd_op.matmul_grad", 129, input_types=[self.t_dtensor([16, 2, 96], self.t_f32()), self.t_dtensor([96, 32], self.t_f32()), self.t_dtensor([16, 2, 32], self.t_f32())], output_types=[self.t_null(), self.t_dtensor([96, 32], self.t_f32())], attrs={"stop_gradient":self.a_array(self.a_bool(False), self.a_bool(False)), "transpose_x":self.a_bool(False), "transpose_y":self.a_bool(False), "__operands_symbols_signature__":self.a_array(self.a_symbol(self.s_null()), self.a_symbol(self.s_null()), self.a_symbol(self.s_null())), "__results_symbols_signature__":self.a_array(self.a_symbol(self.s_null()), self.a_symbol(self.s_null()))})

    self.shadow_output_130 = self.Op("builtin.shadow_output", 130, input_types=[self.t_dtensor([32], self.t_f32())], output_types=[], attrs={"output_name":self.a_str("param_grad_0"), "__operands_symbols_signature__":self.a_array(self.a_symbol(self.s_null())), "__results_symbols_signature__":self.a_array()})

    self.shadow_output_131 = self.Op("builtin.shadow_output", 131, input_types=[self.t_dtensor([96, 32], self.t_f32())], output_types=[], attrs={"output_name":self.a_str("param_grad_1"), "__operands_symbols_signature__":self.a_array(self.a_symbol(self.s_null())), "__results_symbols_signature__":self.a_array()})

    self.shadow_output_132 = self.Op("builtin.shadow_output", 132, input_types=[self.t_dtensor([16], self.t_f32())], output_types=[], attrs={"output_name":self.a_str("param_grad_2"), "__operands_symbols_signature__":self.a_array(self.a_symbol(self.s_null())), "__results_symbols_signature__":self.a_array()})

    self.shadow_output_133 = self.Op("builtin.shadow_output", 133, input_types=[self.t_dtensor([32, 16], self.t_f32())], output_types=[], attrs={"output_name":self.a_str("param_grad_3"), "__operands_symbols_signature__":self.a_array(self.a_symbol(self.s_null())), "__results_symbols_signature__":self.a_array()})

    self.module_104 = self.Op("builtin.module", 104, input_types=[], output_types=[], attrs={"program":self.a_pointer("0x584e2360"), "__operands_symbols_signature__":self.a_array(), "__results_symbols_signature__":self.a_array()}, block_positional_arg_names=[[[]]], block_keyword_arg_names=[[{"output_grad_0": "arg_1481526688", "middle_5": "arg_1497418592", "middle_4": "arg_1497418384", "middle_2": "arg_1481508288", "middle_0": "arg_1481527936", "linear_1.w_0": "arg_1457990560", "linear_1.b_0": "arg_1457989680", "middle_3": "arg_1481508496", "middle_1": "arg_1481528144", "linear_0.w_0": "arg_1497436400", "linear_0.b_0": "arg_1457913760", "args_0": "arg_1497471536"}]], block_positional_arg_types=[[[]]], block_keyword_arg_types=[[[self.t_dtensor([16, 2, 16], self.t_f32()), self.t_dtensor([16, 2, 16], self.t_f32()), self.t_dtensor([16, 2, 32], self.t_ui8()), self.t_dtensor([1], self.t_f32()), self.t_dtensor([16, 2, 32], self.t_f32()), self.t_dtensor([32, 16], self.t_f32()), self.t_dtensor([16], self.t_f32()), self.t_dtensor([16, 2, 32], self.t_f32()), self.t_dtensor([16, 2, 32], self.t_f32()), self.t_dtensor([96, 32], self.t_f32()), self.t_dtensor([32], self.t_f32()), self.t_dtensor([16, 2, 96], self.t_f32())]]], )

    

  def module_104_block00(self, call):

    def ret_lambda_module_104_block00(arg_1481526688, arg_1497418592, arg_1497418384, arg_1481508288, arg_1481527936, arg_1457990560, arg_1457989680, arg_1481508496, arg_1481528144, arg_1497436400, arg_1457913760, arg_1497471536):

      add_grad_1240, add_grad_1241, = call(self.add_grad_124, arg_1497418592, arg_1457989680, arg_1481526688)

      matmul_grad_1250, matmul_grad_1251, = call(self.matmul_grad_125, arg_1481508496, arg_1457990560, add_grad_1240)

      dropout_grad_1260, = call(self.dropout_grad_126, arg_1497418384, matmul_grad_1250, arg_1481508288)

      relu_grad_1270, = call(self.relu_grad_127, arg_1481528144, dropout_grad_1260)

      add_grad_1280, add_grad_1281, = call(self.add_grad_128, arg_1481527936, arg_1457913760, relu_grad_1270)

      matmul_grad_1290, matmul_grad_1291, = call(self.matmul_grad_129, arg_1497471536, arg_1497436400, add_grad_1280)

      call(self.shadow_output_130, add_grad_1281)

      call(self.shadow_output_131, matmul_grad_1291)

      call(self.shadow_output_132, add_grad_1241)

      call(self.shadow_output_133, matmul_grad_1251)

    return ret_lambda_module_104_block00

    

  def __call__(self, call, *args, **kwargs):

    self.SetArgs(args)

    self.SetKeywordArgs(kwargs)

    return call(self.module_104, blocks=[[(self.module_104_block00,)]])


class PirProgram_5426476789613677471:

  def __init__(self):

    self.parameter_459 = self.Op("builtin.parameter", 459, input_types=[], output_types=[self.t_dtensor([16], self.t_f32())], attrs={"is_parameter":self.a_array(self.a_bool(True)), "is_distributed":self.a_array(self.a_bool(False)), "need_clip":self.a_array(self.a_bool(True)), "persistable":self.a_array(self.a_bool(True)), "stop_gradient":self.a_array(self.a_bool(False)), "trainable":self.a_array(self.a_bool(True)), "parameter_name":self.a_str("linear_1.b_0"), "__operands_symbols_signature__":self.a_array(), "__results_symbols_signature__":self.a_array(self.a_symbol(self.s_null()))})

    self.parameter_460 = self.Op("builtin.parameter", 460, input_types=[], output_types=[self.t_dtensor([32, 16], self.t_f32())], attrs={"is_parameter":self.a_array(self.a_bool(True)), "is_distributed":self.a_array(self.a_bool(False)), "need_clip":self.a_array(self.a_bool(True)), "persistable":self.a_array(self.a_bool(True)), "stop_gradient":self.a_array(self.a_bool(False)), "trainable":self.a_array(self.a_bool(True)), "parameter_name":self.a_str("linear_1.w_0"), "__operands_symbols_signature__":self.a_array(), "__results_symbols_signature__":self.a_array(self.a_symbol(self.s_null()))})

    self.parameter_461 = self.Op("builtin.parameter", 461, input_types=[], output_types=[self.t_dtensor([32], self.t_f32())], attrs={"is_parameter":self.a_array(self.a_bool(True)), "is_distributed":self.a_array(self.a_bool(False)), "need_clip":self.a_array(self.a_bool(True)), "persistable":self.a_array(self.a_bool(True)), "stop_gradient":self.a_array(self.a_bool(False)), "trainable":self.a_array(self.a_bool(True)), "parameter_name":self.a_str("linear_0.b_0"), "__operands_symbols_signature__":self.a_array(), "__results_symbols_signature__":self.a_array(self.a_symbol(self.s_null()))})

    self.parameter_462 = self.Op("builtin.parameter", 462, input_types=[], output_types=[self.t_dtensor([96, 32], self.t_f32())], attrs={"is_parameter":self.a_array(self.a_bool(True)), "is_distributed":self.a_array(self.a_bool(False)), "need_clip":self.a_array(self.a_bool(True)), "struct_name":self.a_str("/Linear_4/"), "persistable":self.a_array(self.a_bool(True)), "stop_gradient":self.a_array(self.a_bool(False)), "trainable":self.a_array(self.a_bool(True)), "parameter_name":self.a_str("linear_0.w_0"), "__operands_symbols_signature__":self.a_array(), "__results_symbols_signature__":self.a_array(self.a_symbol(self.s_null()))})

    self.data_463 = self.Op("pd_op.data", 463, input_types=[], output_types=[self.t_dtensor([-1, 2, -1], self.t_f32())], attrs={"struct_name":self.a_str("/Linear_4/"), "stop_gradient":self.a_array(self.a_bool(True)), "name":self.a_str("args_0"), "shape":self.a_intarray(-1, 2, -1), "dtype":self.a_dtype("float32"), "place":self.a_place("undefined", 0), "__operands_symbols_signature__":self.a_array(), "__results_symbols_signature__":self.a_array(self.a_symbol(self.s_null()))})

    self.matmul_464 = self.Op("pd_op.matmul", 464, input_types=[self.t_dtensor([-1, 2, -1], self.t_f32()), self.t_dtensor([96, 32], self.t_f32())], output_types=[self.t_dtensor([-1, 2, 32], self.t_f32())], attrs={"struct_name":self.a_str("/Linear_4/"), "stop_gradient":self.a_array(self.a_bool(False)), "transpose_x":self.a_bool(False), "transpose_y":self.a_bool(False), "__operands_symbols_signature__":self.a_array(self.a_symbol(self.s_null()), self.a_symbol(self.s_null())), "__results_symbols_signature__":self.a_array(self.a_symbol(self.s_null()))})

    self.add_465 = self.Op("pd_op.add", 465, input_types=[self.t_dtensor([-1, 2, 32], self.t_f32()), self.t_dtensor([32], self.t_f32())], output_types=[self.t_dtensor([-1, 2, 32], self.t_f32())], attrs={"struct_name":self.a_str("/Linear_4/"), "stop_gradient":self.a_array(self.a_bool(False)), "__operands_symbols_signature__":self.a_array(self.a_symbol(self.s_null()), self.a_symbol(self.s_null())), "__results_symbols_signature__":self.a_array(self.a_symbol(self.s_null()))})

    self.relu_466 = self.Op("pd_op.relu", 466, input_types=[self.t_dtensor([-1, 2, 32], self.t_f32())], output_types=[self.t_dtensor([-1, 2, 32], self.t_f32())], attrs={"struct_name":self.a_str("/ReLU_2/"), "stop_gradient":self.a_array(self.a_bool(False)), "__operands_symbols_signature__":self.a_array(self.a_symbol(self.s_null())), "__results_symbols_signature__":self.a_array(self.a_symbol(self.s_null()))})

    self.full_467 = self.Op("pd_op.full", 467, input_types=[], output_types=[self.t_dtensor([1], self.t_f32())], attrs={"struct_name":self.a_str("/Dropout_2/"), "stop_gradient":self.a_array(self.a_bool(True)), "shape":self.a_intarray(1), "value":self.a_f64("0.2"), "dtype":self.a_dtype("float32"), "place":self.a_place("cpu"), "__operands_symbols_signature__":self.a_array(), "__results_symbols_signature__":self.a_array(self.a_symbol(self.s_null()))})

    self.dropout_468 = self.Op("pd_op.dropout", 468, input_types=[self.t_dtensor([-1, 2, 32], self.t_f32()), self.t_null(), self.t_dtensor([1], self.t_f32())], output_types=[self.t_dtensor([-1, 2, 32], self.t_f32()), self.t_dtensor([-1, 2, 32], self.t_ui8())], attrs={"struct_name":self.a_str("/Dropout_2/"), "stop_gradient":self.a_array(self.a_bool(False), self.a_bool(False)), "is_test":self.a_bool(False), "seed":self.a_i32(0), "mode":self.a_str("upscale_in_train"), "fix_seed":self.a_bool(False), "__operands_symbols_signature__":self.a_array(self.a_symbol(self.s_null()), self.a_symbol(self.s_null()), self.a_symbol(self.s_null())), "__results_symbols_signature__":self.a_array(self.a_symbol(self.s_null()), self.a_symbol(self.s_null()))})

    self.matmul_469 = self.Op("pd_op.matmul", 469, input_types=[self.t_dtensor([-1, 2, 32], self.t_f32()), self.t_dtensor([32, 16], self.t_f32())], output_types=[self.t_dtensor([-1, 2, 16], self.t_f32())], attrs={"struct_name":self.a_str("/Linear_5/"), "stop_gradient":self.a_array(self.a_bool(False)), "transpose_x":self.a_bool(False), "transpose_y":self.a_bool(False), "__operands_symbols_signature__":self.a_array(self.a_symbol(self.s_null()), self.a_symbol(self.s_null())), "__results_symbols_signature__":self.a_array(self.a_symbol(self.s_null()))})

    self.add_470 = self.Op("pd_op.add", 470, input_types=[self.t_dtensor([-1, 2, 16], self.t_f32()), self.t_dtensor([16], self.t_f32())], output_types=[self.t_dtensor([-1, 2, 16], self.t_f32())], attrs={"struct_name":self.a_str("/Linear_5/"), "stop_gradient":self.a_array(self.a_bool(False)), "__operands_symbols_signature__":self.a_array(self.a_symbol(self.s_null()), self.a_symbol(self.s_null())), "__results_symbols_signature__":self.a_array(self.a_symbol(self.s_null()))})

    self.shadow_output_471 = self.Op("builtin.shadow_output", 471, input_types=[self.t_dtensor([-1, 2, 32], self.t_f32())], output_types=[], attrs={"output_name":self.a_str("middle_0"), "__operands_symbols_signature__":self.a_array(self.a_symbol(self.s_null())), "__results_symbols_signature__":self.a_array()})

    self.shadow_output_472 = self.Op("builtin.shadow_output", 472, input_types=[self.t_dtensor([-1, 2, 32], self.t_f32())], output_types=[], attrs={"output_name":self.a_str("middle_1"), "__operands_symbols_signature__":self.a_array(self.a_symbol(self.s_null())), "__results_symbols_signature__":self.a_array()})

    self.shadow_output_473 = self.Op("builtin.shadow_output", 473, input_types=[self.t_dtensor([1], self.t_f32())], output_types=[], attrs={"output_name":self.a_str("middle_2"), "__operands_symbols_signature__":self.a_array(self.a_symbol(self.s_null())), "__results_symbols_signature__":self.a_array()})

    self.shadow_output_474 = self.Op("builtin.shadow_output", 474, input_types=[self.t_dtensor([-1, 2, 32], self.t_f32())], output_types=[], attrs={"output_name":self.a_str("middle_3"), "__operands_symbols_signature__":self.a_array(self.a_symbol(self.s_null())), "__results_symbols_signature__":self.a_array()})

    self.shadow_output_475 = self.Op("builtin.shadow_output", 475, input_types=[self.t_dtensor([-1, 2, 32], self.t_ui8())], output_types=[], attrs={"output_name":self.a_str("middle_4"), "__operands_symbols_signature__":self.a_array(self.a_symbol(self.s_null())), "__results_symbols_signature__":self.a_array()})

    self.shadow_output_476 = self.Op("builtin.shadow_output", 476, input_types=[self.t_dtensor([-1, 2, 16], self.t_f32())], output_types=[], attrs={"output_name":self.a_str("middle_5"), "__operands_symbols_signature__":self.a_array(self.a_symbol(self.s_null())), "__results_symbols_signature__":self.a_array()})

    self.shadow_output_477 = self.Op("builtin.shadow_output", 477, input_types=[self.t_dtensor([-1, 2, 16], self.t_f32())], output_types=[], attrs={"output_name":self.a_str("output_0"), "__operands_symbols_signature__":self.a_array(self.a_symbol(self.s_null())), "__results_symbols_signature__":self.a_array()})

    self.module_457 = self.Op("builtin.module", 457, input_types=[], output_types=[], attrs={"program":self.a_pointer("0x6b3cf350"), "__operands_symbols_signature__":self.a_array(), "__results_symbols_signature__":self.a_array()}, block_positional_arg_names=[[[]]], block_keyword_arg_names=[[{}]], block_positional_arg_types=[[[]]], block_keyword_arg_types=[[[]]], )

    

  def module_457_block00(self, call):

    def ret_lambda_module_457_block00():

      parameter_4590, = call(self.parameter_459)

      parameter_4600, = call(self.parameter_460)

      parameter_4610, = call(self.parameter_461)

      parameter_4620, = call(self.parameter_462)

      data_4630, = call(self.data_463)

      matmul_4640, = call(self.matmul_464, data_4630, parameter_4620)

      add_4650, = call(self.add_465, matmul_4640, parameter_4610)

      relu_4660, = call(self.relu_466, add_4650)

      full_4670, = call(self.full_467)

      dropout_4680, dropout_4681, = call(self.dropout_468, relu_4660, None, full_4670)

      matmul_4690, = call(self.matmul_469, dropout_4680, parameter_4600)

      add_4700, = call(self.add_470, matmul_4690, parameter_4590)

      call(self.shadow_output_471, matmul_4640)

      call(self.shadow_output_472, relu_4660)

      call(self.shadow_output_473, full_4670)

      call(self.shadow_output_474, dropout_4680)

      call(self.shadow_output_475, dropout_4681)

      call(self.shadow_output_476, matmul_4690)

      call(self.shadow_output_477, add_4700)

    return ret_lambda_module_457_block00

    

  def __call__(self, call, *args, **kwargs):

    self.SetArgs(args)

    self.SetKeywordArgs(kwargs)

    return call(self.module_457, blocks=[[(self.module_457_block00,)]])


class PirProgram_7028096434133672773:

  def __init__(self):

    self.parameter_613 = self.Op("builtin.parameter", 613, input_types=[], output_types=[self.t_dtensor([96], self.t_f32())], attrs={"is_parameter":self.a_array(self.a_bool(True)), "is_distributed":self.a_array(self.a_bool(False)), "need_clip":self.a_array(self.a_bool(True)), "persistable":self.a_array(self.a_bool(True)), "stop_gradient":self.a_array(self.a_bool(False)), "trainable":self.a_array(self.a_bool(True)), "parameter_name":self.a_str("linear_3.b_0"), "__operands_symbols_signature__":self.a_array(), "__results_symbols_signature__":self.a_array(self.a_symbol(self.s_null()))})

    self.parameter_614 = self.Op("builtin.parameter", 614, input_types=[], output_types=[self.t_dtensor([32, 96], self.t_f32())], attrs={"is_parameter":self.a_array(self.a_bool(True)), "is_distributed":self.a_array(self.a_bool(False)), "need_clip":self.a_array(self.a_bool(True)), "persistable":self.a_array(self.a_bool(True)), "stop_gradient":self.a_array(self.a_bool(False)), "trainable":self.a_array(self.a_bool(True)), "parameter_name":self.a_str("linear_3.w_0"), "__operands_symbols_signature__":self.a_array(), "__results_symbols_signature__":self.a_array(self.a_symbol(self.s_null()))})

    self.parameter_615 = self.Op("builtin.parameter", 615, input_types=[], output_types=[self.t_dtensor([32], self.t_f32())], attrs={"is_parameter":self.a_array(self.a_bool(True)), "is_distributed":self.a_array(self.a_bool(False)), "need_clip":self.a_array(self.a_bool(True)), "persistable":self.a_array(self.a_bool(True)), "stop_gradient":self.a_array(self.a_bool(False)), "trainable":self.a_array(self.a_bool(True)), "parameter_name":self.a_str("linear_2.b_0"), "__operands_symbols_signature__":self.a_array(), "__results_symbols_signature__":self.a_array(self.a_symbol(self.s_null()))})

    self.parameter_616 = self.Op("builtin.parameter", 616, input_types=[], output_types=[self.t_dtensor([16, 32], self.t_f32())], attrs={"is_parameter":self.a_array(self.a_bool(True)), "is_distributed":self.a_array(self.a_bool(False)), "need_clip":self.a_array(self.a_bool(True)), "struct_name":self.a_str("/Linear_6/"), "persistable":self.a_array(self.a_bool(True)), "stop_gradient":self.a_array(self.a_bool(False)), "trainable":self.a_array(self.a_bool(True)), "parameter_name":self.a_str("linear_2.w_0"), "__operands_symbols_signature__":self.a_array(), "__results_symbols_signature__":self.a_array(self.a_symbol(self.s_null()))})

    self.data_617 = self.Op("pd_op.data", 617, input_types=[], output_types=[self.t_dtensor([-1, 2, -1], self.t_f32())], attrs={"struct_name":self.a_str("/Linear_6/"), "stop_gradient":self.a_array(self.a_bool(False)), "name":self.a_str("args_0"), "shape":self.a_intarray(-1, 2, -1), "dtype":self.a_dtype("float32"), "place":self.a_place("undefined", 0), "__operands_symbols_signature__":self.a_array(), "__results_symbols_signature__":self.a_array(self.a_symbol(self.s_null()))})

    self.matmul_618 = self.Op("pd_op.matmul", 618, input_types=[self.t_dtensor([-1, 2, -1], self.t_f32()), self.t_dtensor([16, 32], self.t_f32())], output_types=[self.t_dtensor([-1, 2, 32], self.t_f32())], attrs={"struct_name":self.a_str("/Linear_6/"), "stop_gradient":self.a_array(self.a_bool(False)), "transpose_x":self.a_bool(False), "transpose_y":self.a_bool(False), "__operands_symbols_signature__":self.a_array(self.a_symbol(self.s_null()), self.a_symbol(self.s_null())), "__results_symbols_signature__":self.a_array(self.a_symbol(self.s_null()))})

    self.add_619 = self.Op("pd_op.add", 619, input_types=[self.t_dtensor([-1, 2, 32], self.t_f32()), self.t_dtensor([32], self.t_f32())], output_types=[self.t_dtensor([-1, 2, 32], self.t_f32())], attrs={"struct_name":self.a_str("/Linear_6/"), "stop_gradient":self.a_array(self.a_bool(False)), "__operands_symbols_signature__":self.a_array(self.a_symbol(self.s_null()), self.a_symbol(self.s_null())), "__results_symbols_signature__":self.a_array(self.a_symbol(self.s_null()))})

    self.relu_620 = self.Op("pd_op.relu", 620, input_types=[self.t_dtensor([-1, 2, 32], self.t_f32())], output_types=[self.t_dtensor([-1, 2, 32], self.t_f32())], attrs={"struct_name":self.a_str("/ReLU_3/"), "stop_gradient":self.a_array(self.a_bool(False)), "__operands_symbols_signature__":self.a_array(self.a_symbol(self.s_null())), "__results_symbols_signature__":self.a_array(self.a_symbol(self.s_null()))})

    self.full_621 = self.Op("pd_op.full", 621, input_types=[], output_types=[self.t_dtensor([1], self.t_f32())], attrs={"struct_name":self.a_str("/Dropout_3/"), "stop_gradient":self.a_array(self.a_bool(True)), "shape":self.a_intarray(1), "value":self.a_f64("0.2"), "dtype":self.a_dtype("float32"), "place":self.a_place("cpu"), "__operands_symbols_signature__":self.a_array(), "__results_symbols_signature__":self.a_array(self.a_symbol(self.s_null()))})

    self.dropout_622 = self.Op("pd_op.dropout", 622, input_types=[self.t_dtensor([-1, 2, 32], self.t_f32()), self.t_null(), self.t_dtensor([1], self.t_f32())], output_types=[self.t_dtensor([-1, 2, 32], self.t_f32()), self.t_dtensor([-1, 2, 32], self.t_ui8())], attrs={"struct_name":self.a_str("/Dropout_3/"), "stop_gradient":self.a_array(self.a_bool(False), self.a_bool(False)), "is_test":self.a_bool(False), "seed":self.a_i32(0), "mode":self.a_str("upscale_in_train"), "fix_seed":self.a_bool(False), "__operands_symbols_signature__":self.a_array(self.a_symbol(self.s_null()), self.a_symbol(self.s_null()), self.a_symbol(self.s_null())), "__results_symbols_signature__":self.a_array(self.a_symbol(self.s_null()), self.a_symbol(self.s_null()))})

    self.matmul_623 = self.Op("pd_op.matmul", 623, input_types=[self.t_dtensor([-1, 2, 32], self.t_f32()), self.t_dtensor([32, 96], self.t_f32())], output_types=[self.t_dtensor([-1, 2, 96], self.t_f32())], attrs={"struct_name":self.a_str("/Linear_7/"), "stop_gradient":self.a_array(self.a_bool(False)), "transpose_x":self.a_bool(False), "transpose_y":self.a_bool(False), "__operands_symbols_signature__":self.a_array(self.a_symbol(self.s_null()), self.a_symbol(self.s_null())), "__results_symbols_signature__":self.a_array(self.a_symbol(self.s_null()))})

    self.add_624 = self.Op("pd_op.add", 624, input_types=[self.t_dtensor([-1, 2, 96], self.t_f32()), self.t_dtensor([96], self.t_f32())], output_types=[self.t_dtensor([-1, 2, 96], self.t_f32())], attrs={"struct_name":self.a_str("/Linear_7/"), "stop_gradient":self.a_array(self.a_bool(False)), "__operands_symbols_signature__":self.a_array(self.a_symbol(self.s_null()), self.a_symbol(self.s_null())), "__results_symbols_signature__":self.a_array(self.a_symbol(self.s_null()))})

    self.shadow_output_625 = self.Op("builtin.shadow_output", 625, input_types=[self.t_dtensor([-1, 2, 32], self.t_f32())], output_types=[], attrs={"output_name":self.a_str("middle_0"), "__operands_symbols_signature__":self.a_array(self.a_symbol(self.s_null())), "__results_symbols_signature__":self.a_array()})

    self.shadow_output_626 = self.Op("builtin.shadow_output", 626, input_types=[self.t_dtensor([-1, 2, 32], self.t_f32())], output_types=[], attrs={"output_name":self.a_str("middle_1"), "__operands_symbols_signature__":self.a_array(self.a_symbol(self.s_null())), "__results_symbols_signature__":self.a_array()})

    self.shadow_output_627 = self.Op("builtin.shadow_output", 627, input_types=[self.t_dtensor([1], self.t_f32())], output_types=[], attrs={"output_name":self.a_str("middle_2"), "__operands_symbols_signature__":self.a_array(self.a_symbol(self.s_null())), "__results_symbols_signature__":self.a_array()})

    self.shadow_output_628 = self.Op("builtin.shadow_output", 628, input_types=[self.t_dtensor([-1, 2, 32], self.t_f32())], output_types=[], attrs={"output_name":self.a_str("middle_3"), "__operands_symbols_signature__":self.a_array(self.a_symbol(self.s_null())), "__results_symbols_signature__":self.a_array()})

    self.shadow_output_629 = self.Op("builtin.shadow_output", 629, input_types=[self.t_dtensor([-1, 2, 32], self.t_ui8())], output_types=[], attrs={"output_name":self.a_str("middle_4"), "__operands_symbols_signature__":self.a_array(self.a_symbol(self.s_null())), "__results_symbols_signature__":self.a_array()})

    self.shadow_output_630 = self.Op("builtin.shadow_output", 630, input_types=[self.t_dtensor([-1, 2, 96], self.t_f32())], output_types=[], attrs={"output_name":self.a_str("middle_5"), "__operands_symbols_signature__":self.a_array(self.a_symbol(self.s_null())), "__results_symbols_signature__":self.a_array()})

    self.shadow_output_631 = self.Op("builtin.shadow_output", 631, input_types=[self.t_dtensor([-1, 2, 96], self.t_f32())], output_types=[], attrs={"output_name":self.a_str("output_0"), "__operands_symbols_signature__":self.a_array(self.a_symbol(self.s_null())), "__results_symbols_signature__":self.a_array()})

    self.module_611 = self.Op("builtin.module", 611, input_types=[], output_types=[], attrs={"program":self.a_pointer("0x6b400f50"), "__operands_symbols_signature__":self.a_array(), "__results_symbols_signature__":self.a_array()}, block_positional_arg_names=[[[]]], block_keyword_arg_names=[[{}]], block_positional_arg_types=[[[]]], block_keyword_arg_types=[[[]]], )

    

  def module_611_block00(self, call):

    def ret_lambda_module_611_block00():

      parameter_6130, = call(self.parameter_613)

      parameter_6140, = call(self.parameter_614)

      parameter_6150, = call(self.parameter_615)

      parameter_6160, = call(self.parameter_616)

      data_6170, = call(self.data_617)

      matmul_6180, = call(self.matmul_618, data_6170, parameter_6160)

      add_6190, = call(self.add_619, matmul_6180, parameter_6150)

      relu_6200, = call(self.relu_620, add_6190)

      full_6210, = call(self.full_621)

      dropout_6220, dropout_6221, = call(self.dropout_622, relu_6200, None, full_6210)

      matmul_6230, = call(self.matmul_623, dropout_6220, parameter_6140)

      add_6240, = call(self.add_624, matmul_6230, parameter_6130)

      call(self.shadow_output_625, matmul_6180)

      call(self.shadow_output_626, relu_6200)

      call(self.shadow_output_627, full_6210)

      call(self.shadow_output_628, dropout_6220)

      call(self.shadow_output_629, dropout_6221)

      call(self.shadow_output_630, matmul_6230)

      call(self.shadow_output_631, add_6240)

    return ret_lambda_module_611_block00

    

  def __call__(self, call, *args, **kwargs):

    self.SetArgs(args)

    self.SetKeywordArgs(kwargs)

    return call(self.module_611, blocks=[[(self.module_611_block00,)]])


class PirProgram_3000443918524221177:

  def __init__(self):

    self.add_grad_632 = self.Op("pd_op.add_grad", 632, input_types=[self.t_dtensor([-1, 2, 96], self.t_f32()), self.t_dtensor([96], self.t_f32()), self.t_dtensor([-1, 2, 96], self.t_f32())], output_types=[self.t_dtensor([-1, 2, 96], self.t_f32()), self.t_dtensor([96], self.t_f32())], attrs={"stop_gradient":self.a_array(self.a_bool(False), self.a_bool(False)), "axis":self.a_i32(-1), "__operands_symbols_signature__":self.a_array(self.a_symbol(self.s_null()), self.a_symbol(self.s_null()), self.a_symbol(self.s_null())), "__results_symbols_signature__":self.a_array(self.a_symbol(self.s_null()), self.a_symbol(self.s_null()))})

    self.matmul_grad_633 = self.Op("pd_op.matmul_grad", 633, input_types=[self.t_dtensor([-1, 2, 32], self.t_f32()), self.t_dtensor([32, 96], self.t_f32()), self.t_dtensor([-1, 2, 96], self.t_f32())], output_types=[self.t_dtensor([-1, 2, 32], self.t_f32()), self.t_dtensor([32, 96], self.t_f32())], attrs={"stop_gradient":self.a_array(self.a_bool(False), self.a_bool(False)), "transpose_x":self.a_bool(False), "transpose_y":self.a_bool(False), "__operands_symbols_signature__":self.a_array(self.a_symbol(self.s_null()), self.a_symbol(self.s_null()), self.a_symbol(self.s_null())), "__results_symbols_signature__":self.a_array(self.a_symbol(self.s_null()), self.a_symbol(self.s_null()))})

    self.dropout_grad_634 = self.Op("pd_op.dropout_grad", 634, input_types=[self.t_dtensor([-1, 2, 32], self.t_ui8()), self.t_dtensor([-1, 2, 32], self.t_f32()), self.t_dtensor([1], self.t_f32())], output_types=[self.t_dtensor([-1, 2, 32], self.t_f32())], attrs={"stop_gradient":self.a_array(self.a_bool(False)), "is_test":self.a_bool(False), "mode":self.a_str("upscale_in_train"), "__operands_symbols_signature__":self.a_array(self.a_symbol(self.s_null()), self.a_symbol(self.s_null()), self.a_symbol(self.s_null())), "__results_symbols_signature__":self.a_array(self.a_symbol(self.s_null()))})

    self.relu_grad_635 = self.Op("pd_op.relu_grad", 635, input_types=[self.t_dtensor([-1, 2, 32], self.t_f32()), self.t_dtensor([-1, 2, 32], self.t_f32())], output_types=[self.t_dtensor([-1, 2, 32], self.t_f32())], attrs={"stop_gradient":self.a_array(self.a_bool(False)), "__operands_symbols_signature__":self.a_array(self.a_symbol(self.s_null()), self.a_symbol(self.s_null())), "__results_symbols_signature__":self.a_array(self.a_symbol(self.s_null()))})

    self.add_grad_636 = self.Op("pd_op.add_grad", 636, input_types=[self.t_dtensor([-1, 2, 32], self.t_f32()), self.t_dtensor([32], self.t_f32()), self.t_dtensor([-1, 2, 32], self.t_f32())], output_types=[self.t_dtensor([-1, 2, 32], self.t_f32()), self.t_dtensor([32], self.t_f32())], attrs={"stop_gradient":self.a_array(self.a_bool(False), self.a_bool(False)), "axis":self.a_i32(-1), "__operands_symbols_signature__":self.a_array(self.a_symbol(self.s_null()), self.a_symbol(self.s_null()), self.a_symbol(self.s_null())), "__results_symbols_signature__":self.a_array(self.a_symbol(self.s_null()), self.a_symbol(self.s_null()))})

    self.matmul_grad_637 = self.Op("pd_op.matmul_grad", 637, input_types=[self.t_dtensor([-1, 2, -1], self.t_f32()), self.t_dtensor([16, 32], self.t_f32()), self.t_dtensor([-1, 2, 32], self.t_f32())], output_types=[self.t_dtensor([-1, 2, -1], self.t_f32()), self.t_dtensor([16, 32], self.t_f32())], attrs={"stop_gradient":self.a_array(self.a_bool(False), self.a_bool(False)), "transpose_x":self.a_bool(False), "transpose_y":self.a_bool(False), "__operands_symbols_signature__":self.a_array(self.a_symbol(self.s_null()), self.a_symbol(self.s_null()), self.a_symbol(self.s_null())), "__results_symbols_signature__":self.a_array(self.a_symbol(self.s_null()), self.a_symbol(self.s_null()))})

    self.shadow_output_638 = self.Op("builtin.shadow_output", 638, input_types=[self.t_dtensor([-1, 2, -1], self.t_f32())], output_types=[], attrs={"output_name":self.a_str("input_grad_0"), "__operands_symbols_signature__":self.a_array(self.a_symbol(self.s_null())), "__results_symbols_signature__":self.a_array()})

    self.shadow_output_639 = self.Op("builtin.shadow_output", 639, input_types=[self.t_dtensor([32], self.t_f32())], output_types=[], attrs={"output_name":self.a_str("param_grad_0"), "__operands_symbols_signature__":self.a_array(self.a_symbol(self.s_null())), "__results_symbols_signature__":self.a_array()})

    self.shadow_output_640 = self.Op("builtin.shadow_output", 640, input_types=[self.t_dtensor([16, 32], self.t_f32())], output_types=[], attrs={"output_name":self.a_str("param_grad_1"), "__operands_symbols_signature__":self.a_array(self.a_symbol(self.s_null())), "__results_symbols_signature__":self.a_array()})

    self.shadow_output_641 = self.Op("builtin.shadow_output", 641, input_types=[self.t_dtensor([96], self.t_f32())], output_types=[], attrs={"output_name":self.a_str("param_grad_2"), "__operands_symbols_signature__":self.a_array(self.a_symbol(self.s_null())), "__results_symbols_signature__":self.a_array()})

    self.shadow_output_642 = self.Op("builtin.shadow_output", 642, input_types=[self.t_dtensor([32, 96], self.t_f32())], output_types=[], attrs={"output_name":self.a_str("param_grad_3"), "__operands_symbols_signature__":self.a_array(self.a_symbol(self.s_null())), "__results_symbols_signature__":self.a_array()})

    self.module_612 = self.Op("builtin.module", 612, input_types=[], output_types=[], attrs={"program":self.a_pointer("0x6b3f4bd0"), "__operands_symbols_signature__":self.a_array(), "__results_symbols_signature__":self.a_array()}, block_positional_arg_names=[[[]]], block_keyword_arg_names=[[{"middle_5": "arg_1799310528", "middle_4": "arg_1799317136", "output_grad_0": "arg_1799310976", "linear_3.w_0": "arg_1799293984", "middle_2": "arg_1799316720", "linear_3.b_0": "arg_1799157344", "middle_3": "arg_1799316928", "middle_1": "arg_1799301824", "linear_2.w_0": "arg_1799170736", "middle_0": "arg_1799301552", "linear_2.b_0": "arg_1799157488", "args_0": "arg_1799179680"}]], block_positional_arg_types=[[[]]], block_keyword_arg_types=[[[self.t_dtensor([-1, 2, 96], self.t_f32()), self.t_dtensor([-1, 2, 32], self.t_ui8()), self.t_dtensor([-1, 2, 96], self.t_f32()), self.t_dtensor([32, 96], self.t_f32()), self.t_dtensor([1], self.t_f32()), self.t_dtensor([96], self.t_f32()), self.t_dtensor([-1, 2, 32], self.t_f32()), self.t_dtensor([-1, 2, 32], self.t_f32()), self.t_dtensor([16, 32], self.t_f32()), self.t_dtensor([-1, 2, 32], self.t_f32()), self.t_dtensor([32], self.t_f32()), self.t_dtensor([-1, 2, -1], self.t_f32())]]], )

    

  def module_612_block00(self, call):

    def ret_lambda_module_612_block00(arg_1799310528, arg_1799317136, arg_1799310976, arg_1799293984, arg_1799316720, arg_1799157344, arg_1799316928, arg_1799301824, arg_1799170736, arg_1799301552, arg_1799157488, arg_1799179680):

      add_grad_6320, add_grad_6321, = call(self.add_grad_632, arg_1799310528, arg_1799157344, arg_1799310976)

      matmul_grad_6330, matmul_grad_6331, = call(self.matmul_grad_633, arg_1799316928, arg_1799293984, add_grad_6320)

      dropout_grad_6340, = call(self.dropout_grad_634, arg_1799317136, matmul_grad_6330, arg_1799316720)

      relu_grad_6350, = call(self.relu_grad_635, arg_1799301824, dropout_grad_6340)

      add_grad_6360, add_grad_6361, = call(self.add_grad_636, arg_1799301552, arg_1799157488, relu_grad_6350)

      matmul_grad_6370, matmul_grad_6371, = call(self.matmul_grad_637, arg_1799179680, arg_1799170736, add_grad_6360)

      call(self.shadow_output_638, matmul_grad_6370)

      call(self.shadow_output_639, add_grad_6361)

      call(self.shadow_output_640, matmul_grad_6371)

      call(self.shadow_output_641, add_grad_6321)

      call(self.shadow_output_642, matmul_grad_6331)

    return ret_lambda_module_612_block00

    

  def __call__(self, call, *args, **kwargs):

    self.SetArgs(args)

    self.SetKeywordArgs(kwargs)

    return call(self.module_612, blocks=[[(self.module_612_block00,)]])


class PirProgram_3111236241086672326:

  def __init__(self):

    self.add_grad_478 = self.Op("pd_op.add_grad", 478, input_types=[self.t_dtensor([-1, 2, 16], self.t_f32()), self.t_dtensor([16], self.t_f32()), self.t_dtensor([-1, 2, 16], self.t_f32())], output_types=[self.t_dtensor([-1, 2, 16], self.t_f32()), self.t_dtensor([16], self.t_f32())], attrs={"stop_gradient":self.a_array(self.a_bool(False), self.a_bool(False)), "axis":self.a_i32(-1), "__operands_symbols_signature__":self.a_array(self.a_symbol(self.s_null()), self.a_symbol(self.s_null()), self.a_symbol(self.s_null())), "__results_symbols_signature__":self.a_array(self.a_symbol(self.s_null()), self.a_symbol(self.s_null()))})

    self.matmul_grad_479 = self.Op("pd_op.matmul_grad", 479, input_types=[self.t_dtensor([-1, 2, 32], self.t_f32()), self.t_dtensor([32, 16], self.t_f32()), self.t_dtensor([-1, 2, 16], self.t_f32())], output_types=[self.t_dtensor([-1, 2, 32], self.t_f32()), self.t_dtensor([32, 16], self.t_f32())], attrs={"stop_gradient":self.a_array(self.a_bool(False), self.a_bool(False)), "transpose_x":self.a_bool(False), "transpose_y":self.a_bool(False), "__operands_symbols_signature__":self.a_array(self.a_symbol(self.s_null()), self.a_symbol(self.s_null()), self.a_symbol(self.s_null())), "__results_symbols_signature__":self.a_array(self.a_symbol(self.s_null()), self.a_symbol(self.s_null()))})

    self.dropout_grad_480 = self.Op("pd_op.dropout_grad", 480, input_types=[self.t_dtensor([-1, 2, 32], self.t_ui8()), self.t_dtensor([-1, 2, 32], self.t_f32()), self.t_dtensor([1], self.t_f32())], output_types=[self.t_dtensor([-1, 2, 32], self.t_f32())], attrs={"stop_gradient":self.a_array(self.a_bool(False)), "is_test":self.a_bool(False), "mode":self.a_str("upscale_in_train"), "__operands_symbols_signature__":self.a_array(self.a_symbol(self.s_null()), self.a_symbol(self.s_null()), self.a_symbol(self.s_null())), "__results_symbols_signature__":self.a_array(self.a_symbol(self.s_null()))})

    self.relu_grad_481 = self.Op("pd_op.relu_grad", 481, input_types=[self.t_dtensor([-1, 2, 32], self.t_f32()), self.t_dtensor([-1, 2, 32], self.t_f32())], output_types=[self.t_dtensor([-1, 2, 32], self.t_f32())], attrs={"stop_gradient":self.a_array(self.a_bool(False)), "__operands_symbols_signature__":self.a_array(self.a_symbol(self.s_null()), self.a_symbol(self.s_null())), "__results_symbols_signature__":self.a_array(self.a_symbol(self.s_null()))})

    self.add_grad_482 = self.Op("pd_op.add_grad", 482, input_types=[self.t_dtensor([-1, 2, 32], self.t_f32()), self.t_dtensor([32], self.t_f32()), self.t_dtensor([-1, 2, 32], self.t_f32())], output_types=[self.t_dtensor([-1, 2, 32], self.t_f32()), self.t_dtensor([32], self.t_f32())], attrs={"stop_gradient":self.a_array(self.a_bool(False), self.a_bool(False)), "axis":self.a_i32(-1), "__operands_symbols_signature__":self.a_array(self.a_symbol(self.s_null()), self.a_symbol(self.s_null()), self.a_symbol(self.s_null())), "__results_symbols_signature__":self.a_array(self.a_symbol(self.s_null()), self.a_symbol(self.s_null()))})

    self.matmul_grad_483 = self.Op("pd_op.matmul_grad", 483, input_types=[self.t_dtensor([-1, 2, -1], self.t_f32()), self.t_dtensor([96, 32], self.t_f32()), self.t_dtensor([-1, 2, 32], self.t_f32())], output_types=[self.t_null(), self.t_dtensor([96, 32], self.t_f32())], attrs={"stop_gradient":self.a_array(self.a_bool(False), self.a_bool(False)), "transpose_x":self.a_bool(False), "transpose_y":self.a_bool(False), "__operands_symbols_signature__":self.a_array(self.a_symbol(self.s_null()), self.a_symbol(self.s_null()), self.a_symbol(self.s_null())), "__results_symbols_signature__":self.a_array(self.a_symbol(self.s_null()), self.a_symbol(self.s_null()))})

    self.shadow_output_484 = self.Op("builtin.shadow_output", 484, input_types=[self.t_dtensor([32], self.t_f32())], output_types=[], attrs={"output_name":self.a_str("param_grad_0"), "__operands_symbols_signature__":self.a_array(self.a_symbol(self.s_null())), "__results_symbols_signature__":self.a_array()})

    self.shadow_output_485 = self.Op("builtin.shadow_output", 485, input_types=[self.t_dtensor([96, 32], self.t_f32())], output_types=[], attrs={"output_name":self.a_str("param_grad_1"), "__operands_symbols_signature__":self.a_array(self.a_symbol(self.s_null())), "__results_symbols_signature__":self.a_array()})

    self.shadow_output_486 = self.Op("builtin.shadow_output", 486, input_types=[self.t_dtensor([16], self.t_f32())], output_types=[], attrs={"output_name":self.a_str("param_grad_2"), "__operands_symbols_signature__":self.a_array(self.a_symbol(self.s_null())), "__results_symbols_signature__":self.a_array()})

    self.shadow_output_487 = self.Op("builtin.shadow_output", 487, input_types=[self.t_dtensor([32, 16], self.t_f32())], output_types=[], attrs={"output_name":self.a_str("param_grad_3"), "__operands_symbols_signature__":self.a_array(self.a_symbol(self.s_null())), "__results_symbols_signature__":self.a_array()})

    self.module_458 = self.Op("builtin.module", 458, input_types=[], output_types=[], attrs={"program":self.a_pointer("0x6b3d0ad0"), "__operands_symbols_signature__":self.a_array(), "__results_symbols_signature__":self.a_array()}, block_positional_arg_names=[[[]]], block_keyword_arg_names=[[{"output_grad_0": "arg_1799168768", "middle_5": "arg_1799149040", "middle_4": "arg_1799148832", "middle_2": "arg_1799159872", "middle_0": "arg_1799136912", "linear_1.w_0": "arg_1735320288", "linear_1.b_0": "arg_1798932704", "middle_3": "arg_1799148624", "middle_1": "arg_1799137056", "linear_0.w_0": "arg_1457571760", "linear_0.b_0": "arg_1798932016", "args_0": "arg_1798507504"}]], block_positional_arg_types=[[[]]], block_keyword_arg_types=[[[self.t_dtensor([-1, 2, 16], self.t_f32()), self.t_dtensor([-1, 2, 16], self.t_f32()), self.t_dtensor([-1, 2, 32], self.t_ui8()), self.t_dtensor([1], self.t_f32()), self.t_dtensor([-1, 2, 32], self.t_f32()), self.t_dtensor([32, 16], self.t_f32()), self.t_dtensor([16], self.t_f32()), self.t_dtensor([-1, 2, 32], self.t_f32()), self.t_dtensor([-1, 2, 32], self.t_f32()), self.t_dtensor([96, 32], self.t_f32()), self.t_dtensor([32], self.t_f32()), self.t_dtensor([-1, 2, -1], self.t_f32())]]], )

    

  def module_458_block00(self, call):

    def ret_lambda_module_458_block00(arg_1799168768, arg_1799149040, arg_1799148832, arg_1799159872, arg_1799136912, arg_1735320288, arg_1798932704, arg_1799148624, arg_1799137056, arg_1457571760, arg_1798932016, arg_1798507504):

      add_grad_4780, add_grad_4781, = call(self.add_grad_478, arg_1799149040, arg_1798932704, arg_1799168768)

      matmul_grad_4790, matmul_grad_4791, = call(self.matmul_grad_479, arg_1799148624, arg_1735320288, add_grad_4780)

      dropout_grad_4800, = call(self.dropout_grad_480, arg_1799148832, matmul_grad_4790, arg_1799159872)

      relu_grad_4810, = call(self.relu_grad_481, arg_1799137056, dropout_grad_4800)

      add_grad_4820, add_grad_4821, = call(self.add_grad_482, arg_1799136912, arg_1798932016, relu_grad_4810)

      matmul_grad_4830, matmul_grad_4831, = call(self.matmul_grad_483, arg_1798507504, arg_1457571760, add_grad_4820)

      call(self.shadow_output_484, add_grad_4821)

      call(self.shadow_output_485, matmul_grad_4831)

      call(self.shadow_output_486, add_grad_4781)

      call(self.shadow_output_487, matmul_grad_4791)

    return ret_lambda_module_458_block00

    

  def __call__(self, call, *args, **kwargs):

    self.SetArgs(args)

    self.SetKeywordArgs(kwargs)

    return call(self.module_458, blocks=[[(self.module_458_block00,)]])


class PirProgram_4630264566612914126:

  def __init__(self):

    self.parameter_750 = self.Op("builtin.parameter", 750, input_types=[], output_types=[self.t_dtensor([16], self.t_f32())], attrs={"is_parameter":self.a_array(self.a_bool(True)), "is_distributed":self.a_array(self.a_bool(False)), "need_clip":self.a_array(self.a_bool(True)), "persistable":self.a_array(self.a_bool(True)), "stop_gradient":self.a_array(self.a_bool(False)), "trainable":self.a_array(self.a_bool(True)), "parameter_name":self.a_str("linear_1.b_0"), "__operands_symbols_signature__":self.a_array(), "__results_symbols_signature__":self.a_array(self.a_symbol(self.s_null()))})

    self.parameter_751 = self.Op("builtin.parameter", 751, input_types=[], output_types=[self.t_dtensor([32, 16], self.t_f32())], attrs={"is_parameter":self.a_array(self.a_bool(True)), "is_distributed":self.a_array(self.a_bool(False)), "need_clip":self.a_array(self.a_bool(True)), "persistable":self.a_array(self.a_bool(True)), "stop_gradient":self.a_array(self.a_bool(False)), "trainable":self.a_array(self.a_bool(True)), "parameter_name":self.a_str("linear_1.w_0"), "__operands_symbols_signature__":self.a_array(), "__results_symbols_signature__":self.a_array(self.a_symbol(self.s_null()))})

    self.parameter_752 = self.Op("builtin.parameter", 752, input_types=[], output_types=[self.t_dtensor([32], self.t_f32())], attrs={"is_parameter":self.a_array(self.a_bool(True)), "is_distributed":self.a_array(self.a_bool(False)), "need_clip":self.a_array(self.a_bool(True)), "persistable":self.a_array(self.a_bool(True)), "stop_gradient":self.a_array(self.a_bool(False)), "trainable":self.a_array(self.a_bool(True)), "parameter_name":self.a_str("linear_0.b_0"), "__operands_symbols_signature__":self.a_array(), "__results_symbols_signature__":self.a_array(self.a_symbol(self.s_null()))})

    self.parameter_753 = self.Op("builtin.parameter", 753, input_types=[], output_types=[self.t_dtensor([96, 32], self.t_f32())], attrs={"is_parameter":self.a_array(self.a_bool(True)), "is_distributed":self.a_array(self.a_bool(False)), "need_clip":self.a_array(self.a_bool(True)), "struct_name":self.a_str("/Linear_8/"), "persistable":self.a_array(self.a_bool(True)), "stop_gradient":self.a_array(self.a_bool(False)), "trainable":self.a_array(self.a_bool(True)), "parameter_name":self.a_str("linear_0.w_0"), "__operands_symbols_signature__":self.a_array(), "__results_symbols_signature__":self.a_array(self.a_symbol(self.s_null()))})

    self.data_754 = self.Op("pd_op.data", 754, input_types=[], output_types=[self.t_dtensor([-1, 2, -1], self.t_f32())], attrs={"struct_name":self.a_str("/Linear_8/"), "stop_gradient":self.a_array(self.a_bool(True)), "name":self.a_str("args_0"), "shape":self.a_intarray(-1, 2, -1), "dtype":self.a_dtype("float32"), "place":self.a_place("undefined", 0), "__operands_symbols_signature__":self.a_array(), "__results_symbols_signature__":self.a_array(self.a_symbol(self.s_null()))})

    self.matmul_755 = self.Op("pd_op.matmul", 755, input_types=[self.t_dtensor([-1, 2, -1], self.t_f32()), self.t_dtensor([96, 32], self.t_f32())], output_types=[self.t_dtensor([-1, 2, 32], self.t_f32())], attrs={"struct_name":self.a_str("/Linear_8/"), "stop_gradient":self.a_array(self.a_bool(False)), "transpose_x":self.a_bool(False), "transpose_y":self.a_bool(False), "__operands_symbols_signature__":self.a_array(self.a_symbol(self.s_null()), self.a_symbol(self.s_null())), "__results_symbols_signature__":self.a_array(self.a_symbol(self.s_null()))})

    self.add_756 = self.Op("pd_op.add", 756, input_types=[self.t_dtensor([-1, 2, 32], self.t_f32()), self.t_dtensor([32], self.t_f32())], output_types=[self.t_dtensor([-1, 2, 32], self.t_f32())], attrs={"struct_name":self.a_str("/Linear_8/"), "stop_gradient":self.a_array(self.a_bool(False)), "__operands_symbols_signature__":self.a_array(self.a_symbol(self.s_null()), self.a_symbol(self.s_null())), "__results_symbols_signature__":self.a_array(self.a_symbol(self.s_null()))})

    self.relu_757 = self.Op("pd_op.relu", 757, input_types=[self.t_dtensor([-1, 2, 32], self.t_f32())], output_types=[self.t_dtensor([-1, 2, 32], self.t_f32())], attrs={"struct_name":self.a_str("/ReLU_4/"), "stop_gradient":self.a_array(self.a_bool(False)), "__operands_symbols_signature__":self.a_array(self.a_symbol(self.s_null())), "__results_symbols_signature__":self.a_array(self.a_symbol(self.s_null()))})

    self.full_758 = self.Op("pd_op.full", 758, input_types=[], output_types=[self.t_dtensor([1], self.t_f32())], attrs={"struct_name":self.a_str("/Dropout_4/"), "stop_gradient":self.a_array(self.a_bool(True)), "shape":self.a_intarray(1), "value":self.a_f64("0.2"), "dtype":self.a_dtype("float32"), "place":self.a_place("cpu"), "__operands_symbols_signature__":self.a_array(), "__results_symbols_signature__":self.a_array(self.a_symbol(self.s_null()))})

    self.dropout_759 = self.Op("pd_op.dropout", 759, input_types=[self.t_dtensor([-1, 2, 32], self.t_f32()), self.t_null(), self.t_dtensor([1], self.t_f32())], output_types=[self.t_dtensor([-1, 2, 32], self.t_f32()), self.t_dtensor([-1, 2, 32], self.t_ui8())], attrs={"struct_name":self.a_str("/Dropout_4/"), "stop_gradient":self.a_array(self.a_bool(False), self.a_bool(False)), "is_test":self.a_bool(True), "seed":self.a_i32(0), "mode":self.a_str("upscale_in_train"), "fix_seed":self.a_bool(False), "__operands_symbols_signature__":self.a_array(self.a_symbol(self.s_null()), self.a_symbol(self.s_null()), self.a_symbol(self.s_null())), "__results_symbols_signature__":self.a_array(self.a_symbol(self.s_null()), self.a_symbol(self.s_null()))})

    self.matmul_760 = self.Op("pd_op.matmul", 760, input_types=[self.t_dtensor([-1, 2, 32], self.t_f32()), self.t_dtensor([32, 16], self.t_f32())], output_types=[self.t_dtensor([-1, 2, 16], self.t_f32())], attrs={"struct_name":self.a_str("/Linear_9/"), "stop_gradient":self.a_array(self.a_bool(False)), "transpose_x":self.a_bool(False), "transpose_y":self.a_bool(False), "__operands_symbols_signature__":self.a_array(self.a_symbol(self.s_null()), self.a_symbol(self.s_null())), "__results_symbols_signature__":self.a_array(self.a_symbol(self.s_null()))})

    self.add_761 = self.Op("pd_op.add", 761, input_types=[self.t_dtensor([-1, 2, 16], self.t_f32()), self.t_dtensor([16], self.t_f32())], output_types=[self.t_dtensor([-1, 2, 16], self.t_f32())], attrs={"struct_name":self.a_str("/Linear_9/"), "stop_gradient":self.a_array(self.a_bool(False)), "__operands_symbols_signature__":self.a_array(self.a_symbol(self.s_null()), self.a_symbol(self.s_null())), "__results_symbols_signature__":self.a_array(self.a_symbol(self.s_null()))})

    self.shadow_output_762 = self.Op("builtin.shadow_output", 762, input_types=[self.t_dtensor([-1, 2, 16], self.t_f32())], output_types=[], attrs={"output_name":self.a_str("output_0"), "__operands_symbols_signature__":self.a_array(self.a_symbol(self.s_null())), "__results_symbols_signature__":self.a_array()})

    self.module_748 = self.Op("builtin.module", 748, input_types=[], output_types=[], attrs={"program":self.a_pointer("0x6b3cafd0"), "__operands_symbols_signature__":self.a_array(), "__results_symbols_signature__":self.a_array()}, block_positional_arg_names=[[[]]], block_keyword_arg_names=[[{}]], block_positional_arg_types=[[[]]], block_keyword_arg_types=[[[]]], )

    

  def module_748_block00(self, call):

    def ret_lambda_module_748_block00():

      parameter_7500, = call(self.parameter_750)

      parameter_7510, = call(self.parameter_751)

      parameter_7520, = call(self.parameter_752)

      parameter_7530, = call(self.parameter_753)

      data_7540, = call(self.data_754)

      matmul_7550, = call(self.matmul_755, data_7540, parameter_7530)

      add_7560, = call(self.add_756, matmul_7550, parameter_7520)

      relu_7570, = call(self.relu_757, add_7560)

      full_7580, = call(self.full_758)

      dropout_7590, dropout_7591, = call(self.dropout_759, relu_7570, None, full_7580)

      matmul_7600, = call(self.matmul_760, dropout_7590, parameter_7510)

      add_7610, = call(self.add_761, matmul_7600, parameter_7500)

      call(self.shadow_output_762, add_7610)

    return ret_lambda_module_748_block00

    

  def __call__(self, call, *args, **kwargs):

    self.SetArgs(args)

    self.SetKeywordArgs(kwargs)

    return call(self.module_748, blocks=[[(self.module_748_block00,)]])


class PirProgram_2498128585497043328:

  def __init__(self):

    self.parameter_817 = self.Op("builtin.parameter", 817, input_types=[], output_types=[self.t_dtensor([96], self.t_f32())], attrs={"is_parameter":self.a_array(self.a_bool(True)), "is_distributed":self.a_array(self.a_bool(False)), "need_clip":self.a_array(self.a_bool(True)), "persistable":self.a_array(self.a_bool(True)), "stop_gradient":self.a_array(self.a_bool(False)), "trainable":self.a_array(self.a_bool(True)), "parameter_name":self.a_str("linear_3.b_0"), "__operands_symbols_signature__":self.a_array(), "__results_symbols_signature__":self.a_array(self.a_symbol(self.s_null()))})

    self.parameter_818 = self.Op("builtin.parameter", 818, input_types=[], output_types=[self.t_dtensor([32, 96], self.t_f32())], attrs={"is_parameter":self.a_array(self.a_bool(True)), "is_distributed":self.a_array(self.a_bool(False)), "need_clip":self.a_array(self.a_bool(True)), "persistable":self.a_array(self.a_bool(True)), "stop_gradient":self.a_array(self.a_bool(False)), "trainable":self.a_array(self.a_bool(True)), "parameter_name":self.a_str("linear_3.w_0"), "__operands_symbols_signature__":self.a_array(), "__results_symbols_signature__":self.a_array(self.a_symbol(self.s_null()))})

    self.parameter_819 = self.Op("builtin.parameter", 819, input_types=[], output_types=[self.t_dtensor([32], self.t_f32())], attrs={"is_parameter":self.a_array(self.a_bool(True)), "is_distributed":self.a_array(self.a_bool(False)), "need_clip":self.a_array(self.a_bool(True)), "persistable":self.a_array(self.a_bool(True)), "stop_gradient":self.a_array(self.a_bool(False)), "trainable":self.a_array(self.a_bool(True)), "parameter_name":self.a_str("linear_2.b_0"), "__operands_symbols_signature__":self.a_array(), "__results_symbols_signature__":self.a_array(self.a_symbol(self.s_null()))})

    self.parameter_820 = self.Op("builtin.parameter", 820, input_types=[], output_types=[self.t_dtensor([16, 32], self.t_f32())], attrs={"is_parameter":self.a_array(self.a_bool(True)), "is_distributed":self.a_array(self.a_bool(False)), "need_clip":self.a_array(self.a_bool(True)), "struct_name":self.a_str("/Linear_10/"), "persistable":self.a_array(self.a_bool(True)), "stop_gradient":self.a_array(self.a_bool(False)), "trainable":self.a_array(self.a_bool(True)), "parameter_name":self.a_str("linear_2.w_0"), "__operands_symbols_signature__":self.a_array(), "__results_symbols_signature__":self.a_array(self.a_symbol(self.s_null()))})

    self.data_821 = self.Op("pd_op.data", 821, input_types=[], output_types=[self.t_dtensor([-1, 2, -1], self.t_f32())], attrs={"struct_name":self.a_str("/Linear_10/"), "stop_gradient":self.a_array(self.a_bool(False)), "name":self.a_str("args_0"), "shape":self.a_intarray(-1, 2, -1), "dtype":self.a_dtype("float32"), "place":self.a_place("undefined", 0), "__operands_symbols_signature__":self.a_array(), "__results_symbols_signature__":self.a_array(self.a_symbol(self.s_null()))})

    self.matmul_822 = self.Op("pd_op.matmul", 822, input_types=[self.t_dtensor([-1, 2, -1], self.t_f32()), self.t_dtensor([16, 32], self.t_f32())], output_types=[self.t_dtensor([-1, 2, 32], self.t_f32())], attrs={"struct_name":self.a_str("/Linear_10/"), "stop_gradient":self.a_array(self.a_bool(False)), "transpose_x":self.a_bool(False), "transpose_y":self.a_bool(False), "__operands_symbols_signature__":self.a_array(self.a_symbol(self.s_null()), self.a_symbol(self.s_null())), "__results_symbols_signature__":self.a_array(self.a_symbol(self.s_null()))})

    self.add_823 = self.Op("pd_op.add", 823, input_types=[self.t_dtensor([-1, 2, 32], self.t_f32()), self.t_dtensor([32], self.t_f32())], output_types=[self.t_dtensor([-1, 2, 32], self.t_f32())], attrs={"struct_name":self.a_str("/Linear_10/"), "stop_gradient":self.a_array(self.a_bool(False)), "__operands_symbols_signature__":self.a_array(self.a_symbol(self.s_null()), self.a_symbol(self.s_null())), "__results_symbols_signature__":self.a_array(self.a_symbol(self.s_null()))})

    self.relu_824 = self.Op("pd_op.relu", 824, input_types=[self.t_dtensor([-1, 2, 32], self.t_f32())], output_types=[self.t_dtensor([-1, 2, 32], self.t_f32())], attrs={"struct_name":self.a_str("/ReLU_5/"), "stop_gradient":self.a_array(self.a_bool(False)), "__operands_symbols_signature__":self.a_array(self.a_symbol(self.s_null())), "__results_symbols_signature__":self.a_array(self.a_symbol(self.s_null()))})

    self.full_825 = self.Op("pd_op.full", 825, input_types=[], output_types=[self.t_dtensor([1], self.t_f32())], attrs={"struct_name":self.a_str("/Dropout_5/"), "stop_gradient":self.a_array(self.a_bool(True)), "shape":self.a_intarray(1), "value":self.a_f64("0.2"), "dtype":self.a_dtype("float32"), "place":self.a_place("cpu"), "__operands_symbols_signature__":self.a_array(), "__results_symbols_signature__":self.a_array(self.a_symbol(self.s_null()))})

    self.dropout_826 = self.Op("pd_op.dropout", 826, input_types=[self.t_dtensor([-1, 2, 32], self.t_f32()), self.t_null(), self.t_dtensor([1], self.t_f32())], output_types=[self.t_dtensor([-1, 2, 32], self.t_f32()), self.t_dtensor([-1, 2, 32], self.t_ui8())], attrs={"struct_name":self.a_str("/Dropout_5/"), "stop_gradient":self.a_array(self.a_bool(False), self.a_bool(False)), "is_test":self.a_bool(True), "seed":self.a_i32(0), "mode":self.a_str("upscale_in_train"), "fix_seed":self.a_bool(False), "__operands_symbols_signature__":self.a_array(self.a_symbol(self.s_null()), self.a_symbol(self.s_null()), self.a_symbol(self.s_null())), "__results_symbols_signature__":self.a_array(self.a_symbol(self.s_null()), self.a_symbol(self.s_null()))})

    self.matmul_827 = self.Op("pd_op.matmul", 827, input_types=[self.t_dtensor([-1, 2, 32], self.t_f32()), self.t_dtensor([32, 96], self.t_f32())], output_types=[self.t_dtensor([-1, 2, 96], self.t_f32())], attrs={"struct_name":self.a_str("/Linear_11/"), "stop_gradient":self.a_array(self.a_bool(False)), "transpose_x":self.a_bool(False), "transpose_y":self.a_bool(False), "__operands_symbols_signature__":self.a_array(self.a_symbol(self.s_null()), self.a_symbol(self.s_null())), "__results_symbols_signature__":self.a_array(self.a_symbol(self.s_null()))})

    self.add_828 = self.Op("pd_op.add", 828, input_types=[self.t_dtensor([-1, 2, 96], self.t_f32()), self.t_dtensor([96], self.t_f32())], output_types=[self.t_dtensor([-1, 2, 96], self.t_f32())], attrs={"struct_name":self.a_str("/Linear_11/"), "stop_gradient":self.a_array(self.a_bool(False)), "__operands_symbols_signature__":self.a_array(self.a_symbol(self.s_null()), self.a_symbol(self.s_null())), "__results_symbols_signature__":self.a_array(self.a_symbol(self.s_null()))})

    self.shadow_output_829 = self.Op("builtin.shadow_output", 829, input_types=[self.t_dtensor([-1, 2, 96], self.t_f32())], output_types=[], attrs={"output_name":self.a_str("output_0"), "__operands_symbols_signature__":self.a_array(self.a_symbol(self.s_null())), "__results_symbols_signature__":self.a_array()})

    self.module_815 = self.Op("builtin.module", 815, input_types=[], output_types=[], attrs={"program":self.a_pointer("0x6bc06ee0"), "__operands_symbols_signature__":self.a_array(), "__results_symbols_signature__":self.a_array()}, block_positional_arg_names=[[[]]], block_keyword_arg_names=[[{}]], block_positional_arg_types=[[[]]], block_keyword_arg_types=[[[]]], )

    

  def module_815_block00(self, call):

    def ret_lambda_module_815_block00():

      parameter_8170, = call(self.parameter_817)

      parameter_8180, = call(self.parameter_818)

      parameter_8190, = call(self.parameter_819)

      parameter_8200, = call(self.parameter_820)

      data_8210, = call(self.data_821)

      matmul_8220, = call(self.matmul_822, data_8210, parameter_8200)

      add_8230, = call(self.add_823, matmul_8220, parameter_8190)

      relu_8240, = call(self.relu_824, add_8230)

      full_8250, = call(self.full_825)

      dropout_8260, dropout_8261, = call(self.dropout_826, relu_8240, None, full_8250)

      matmul_8270, = call(self.matmul_827, dropout_8260, parameter_8180)

      add_8280, = call(self.add_828, matmul_8270, parameter_8170)

      call(self.shadow_output_829, add_8280)

    return ret_lambda_module_815_block00

    

  def __call__(self, call, *args, **kwargs):

    self.SetArgs(args)

    self.SetKeywordArgs(kwargs)

    return call(self.module_815, blocks=[[(self.module_815_block00,)]])


