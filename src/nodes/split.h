/* This file is part of onnx2c.
 *
 * Split node - split input 'data' tensor into equally sized chunks.
 *
 */
namespace toC {

class Split : public Node {
	public:
	Split() {
		op_name = "Split";
		output=data=sections=axis=NULL;
	}

	// input and output
	// std::vector<Tensor *> output;
	const Tensor *output;
	const Tensor *data;
	const Tensor *sections;
	const Tensor *axis;

	std::vector<int64_t> _sections;
	int64_t _axis;


	virtual void parseAttributes( onnx::NodeProto &node ) override {
		for( const auto& a : node.attribute() ) {
			if( a.name() == "split" )
				_sections = parse_attribute_ints(a);
			else if( a.name() == "axis" )
				_axis = parse_attribute_int(a);
			else
				ERROR("Unknown attribute to split");
		}
	}

	virtual void resolve(void) override
	{
		data = inputs[0];
		register_input(data, "data");

		if (inputs.size() > 1) {
			sections = inputs[1];
			register_input(sections, "split");
		}

		if (inputs.size() > 2) {
			axis = inputs[2];
			register_input(axis, "axis");
		}

		// the output tensor
		for (int i = 0; i < sections->data_num_elem(); i++)
		{
			Tensor *t = new Tensor;
			t->data_type = data->data_type;
			t->data_dim = data->data_dim;
			t->data_dim[_axis] = sections->get_data_element(i);
			register_output(t, "output" + std::to_string(i));
		}
	}

	/* Body of the node implementing function */
	virtual void print(std::ostream &dst) const override
	{

		INDT_1 << "/* Split */" << std::endl;
		INDT_1 << "/* axis: " << _axis << " */" << std::endl;

		int64_t s = 0;
		for (int i = 0; i < sections->data_num_elem(); i++) 
		{
			INDT_1 << "/* Split " << std::to_string(i) <<  " of size " << std::to_string(sections->get_data_element(i)) << " */" << std::endl;
			std::string out_idx = std::to_string(i);
			std::string in_idx;

			for (unsigned d = 0; d < data->rank(); d++)
			{
				std::string iv = "i" + std::to_string(d);
				std::string ov = "o" + std::to_string(d);
				if (d == _axis) {
					INDT(d + 1) << "for (unsigned " << iv << "=" << s << ", " << ov << "=0; ";
					dst << ov << "<" << sections->get_data_element(i) << "; ";
				} else {
					INDT(d + 1) << "for (unsigned " << iv << "=" << 0 << ", " << ov << "=0; ";
					dst << ov << "<" << data->data_dim[d] << "; ";
				}
				
				dst << iv << "++, " << ov << "++) {" << std::endl;

				out_idx += "[" + ov + "]";
				in_idx += "[" + iv + "]";
			}
			// Copy over data from input to output
			INDT(data->rank() + 1) << "output" << out_idx << " = data" << in_idx << ";" << std::endl;

			// close loops over output dimensions
			for( unsigned r=data->rank(); r>0; r--) {
				INDT(r) << "}" << std::endl;
			}
			s += sections->get_data_element(i);
		}
	}
};
}

