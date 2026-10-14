use clap::Parser;
use prototype::Cli;

fn main() -> anyhow::Result<()> {
    prototype::run(Cli::parse(), &mut std::io::stdout())
}
