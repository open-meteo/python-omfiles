use pyo3_stub_gen::Result;

fn main() -> Result<()> {
    let stub = omfiles::stub_info()?;
    stub.generate()?;
    Ok(())
}
