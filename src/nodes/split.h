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
		data=sections=axis=NULL;
	}

	// input and output
	std::vector<Tensor *> output;
	const Tensor *data;
	const Tensor *sections;
	const Tensor *axis;

	std::vector<int64_t> _sections;
	int64_t _axis;


	virtual void parseAttributes( onnx::NodeProto &node ) override {
		for( const auto& a : node.attribute() ) {
			LOG(TRACE) << "Parsing attribute " << a.name() << std::endl;
			if( a.name() == "sections" )
				_sections = parse_attribute_ints(a);
			else if( a.name() == "axis" )
				_axis = parse_attribute_int(a);
			else
				ERROR("Unknonw attribute to split");
		}
	}

	virtual void resolve(void) override
	{
		data = inputs[0];
		register_input(data, "data");

		if (inputs.size() > 1) {
			sections = inputs[1];
			register_input(sections, "sections");
		}

		if (inputs.size() > 2) {
			axis = inputs[2];
			register_input(axis, "axis");
		}

		if( sections && sections->isConst == false )
			ERROR("Non-const inputs to Slice not handled");

		// the output tensor
		size_t i = 0;
		for (auto s : _sections)
		{
			Tensor *t = new Tensor;
			t->data_type = data->data_type;
			auto end = i + s;
			for (; i < end; i++) {
				data->get_data_element(i);
			}
			register_output(t, "split");
		}
	}

	/* Body of the node implementing function */
	virtual void print(std::ostream &dst) const override
	{

		INDT_1 << "/* Split */" << std::endl;

	}
};
}

