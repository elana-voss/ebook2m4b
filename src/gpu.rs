//! Which graphics card runs the models: an NVIDIA card (CUDA) or another one (DirectML and Vulkan).
//! NVIDIA's driver brings nvidia-smi; a card it lists is taken as NVIDIA.
//! The hidden --card other (main.rs) takes the other route on any machine, to test it where NVIDIA is found.
use std::process::Command;

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Card {
    Nvidia,
    Other,
}

pub fn card() -> Card {
    match Command::new("nvidia-smi").arg("-L").output() {
        Ok(out)
            if out.status.success() && String::from_utf8_lossy(&out.stdout).contains("GPU ") =>
        {
            Card::Nvidia
        }
        _ => Card::Other,
    }
}

/// The card asked for by the hidden --card ("other"), else the one found.
pub fn chosen(forced: Option<&str>) -> Card {
    match forced {
        Some("other") => Card::Other,
        _ => card(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn other_asked_for_is_taken_whatever_card_is_found() {
        assert_eq!(chosen(Some("other")), Card::Other);
    }
}
