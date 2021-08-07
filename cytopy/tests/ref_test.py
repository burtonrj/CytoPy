from cytopy.flow.ref import create_ref_sample


def test_create_ref_sample(example_populated_experiment):
    create_ref_sample(example_populated_experiment, new_file_name="test ref", sample_size=5000)
    assert example_populated_experiment.get_sample("test ref").data(source="primary").shape[0] == 5000
