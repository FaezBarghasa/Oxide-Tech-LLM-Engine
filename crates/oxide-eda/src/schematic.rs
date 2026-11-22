//! Bidirectional Schematic & Netlist AST.
//!
//! Models electronic components (RefDes, MPN, footprints, electrical ratings),
//! pins, and named electrical nets (VIN, VOUT_3V3, GND, I2C_SDA, USB_DP/USB_DN),
//! supporting KiCad 8/9 S-expression (.kicad_sch) and Flux JSON serialization.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Electronic Component representation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ComponentSymbol {
    pub ref_des: String,       // e.g. "R1", "C3", "U1"
    pub mpn: String,           // e.g. "TPS54302DDCR"
    pub value: String,         // e.g. "10uF", "10k"
    pub footprint: String,     // e.g. "Package_TO_SOT_SMD:SOT-23-6"
    pub pins: Vec<PinEntry>,
}

/// Component Pin representation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PinEntry {
    pub pin_number: String,
    pub pin_name: String,
    pub electrical_type: PinType,
    pub connected_net: Option<String>,
}

/// Electrical Pin Classification for ERC checking.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PinType {
    PowerInput,
    PowerOutput,
    Input,
    Output,
    Bidirectional,
    Passive,
    OpenCollector,
    NoConnect,
}

/// Electrical Net connecting component pins.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ElectricalNet {
    pub name: String,
    pub connected_pins: Vec<(String, String)>, // (ref_des, pin_number)
    pub is_power_rail: bool,
    pub target_voltage_v: Option<f32>,
}

/// Complete Circuit Schematic AST Graph.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct SchematicGraph {
    pub title: String,
    pub components: HashMap<String, ComponentSymbol>,
    pub nets: HashMap<String, ElectricalNet>,
}

impl SchematicGraph {
    #[must_use]
    pub fn new(title: impl Into<String>) -> Self {
        Self {
            title: title.into(),
            components: HashMap::new(),
            nets: HashMap::new(),
        }
    }

    /// Adds a component to the schematic.
    pub fn add_component(&mut self, comp: ComponentSymbol) {
        self.components.insert(comp.ref_des.clone(), comp);
    }

    /// Connects a component pin to a named electrical net.
    pub fn connect_pin(&mut self, ref_des: &str, pin_num: &str, net_name: &str) {
        if let Some(comp) = self.components.get_mut(ref_des) {
            for p in &mut comp.pins {
                if p.pin_number == pin_num {
                    p.connected_net = Some(net_name.to_string());
                }
            }
        }

        let net = self
            .nets
            .entry(net_name.to_string())
            .or_insert_with(|| ElectricalNet {
                name: net_name.to_string(),
                connected_pins: Vec::new(),
                is_power_rail: net_name == "GND"
                    || net_name.starts_with("V")
                    || net_name.starts_with("+"),
                target_voltage_v: None,
            });

        let entry = (ref_des.to_string(), pin_num.to_string());
        if !net.connected_pins.contains(&entry) {
            net.connected_pins.push(entry);
        }
    }

    /// Exports schematic to KiCad 8/9 S-expression (.kicad_sch) string.
    #[must_use]
    pub fn export_kicad_sch(&self) -> String {
        let mut out = String::new();
        out.push_str("(kicad_sch (version 20231120) (generator oxide-eda)\n");
        out.push_str(&format!("  (paper \"A4\") (title \"{}\")\n", self.title));

        for (ref_des, comp) in &self.components {
            out.push_str(&format!(
                "  (symbol (lib_id \"{}\") (at 100 100 0) (unit 1)\n",
                comp.footprint
            ));
            out.push_str(&format!(
                "    (property \"Reference\" \"{}\")\n",
                ref_des
            ));
            out.push_str(&format!("    (property \"Value\" \"{}\")\n", comp.value));
            out.push_str(&format!("    (property \"Footprint\" \"{}\")\n", comp.footprint));
            out.push_str("  )\n");
        }

        out.push_str(")\n");
        out
    }
}
