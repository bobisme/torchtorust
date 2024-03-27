use ndarray::{s, Array1, Array2, Zip};
use onnxruntime::{environment::Environment, ndarray::ArrayViewD, tensor::OrtOwnedTensor};

fn run_xor_model() -> Result<(), Box<dyn std::error::Error>> {
    let environment = Environment::builder().with_name("xor_example").build()?;
    let mut session = environment
        .new_session_builder()?
        .with_model_from_file("xor_model.onnx")?;

    let input_data = Array2::<f32>::from_shape_fn((1, 16), |_| rand::random::<f32>().round());
    let a = input_data.slice(s![.., ..8]);
    let b = input_data.slice(s![.., 8..]);
    let a_row = a.row(0);
    let b_row = b.row(0);
    println!("a: {}", a_row);
    println!("b: {}", b_row);
    let xor = Zip::from(&a_row)
        .and(&b_row)
        .map_collect(|&a, &b| a.round() as i32 ^ b.round() as i32);
    println!("expect: {}", xor);

    // Run the model
    let outputs: Vec<OrtOwnedTensor<f32, _>> = session.run(vec![input_data])?;

    // Process the output
    let output = &outputs[0].map(|&x| x.round() as i32).into_shape(8).unwrap();
    println!("actual: {}", output);

    Ok(())
}

fn main() {
    if let Err(e) = run_xor_model() {
        println!("Failed to run XOR model: {}", e);
    }
}
