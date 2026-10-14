use clap::Parser;
use prototype::Cli;

fn main() -> anyhow::Result<()> {
    prototype::run(Cli::parse())
}
