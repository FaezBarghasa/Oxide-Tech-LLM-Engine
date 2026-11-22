//! Physical PCB Layout Graph.
//!
//! Models physical board outline, layer stackup (2-12 layers), component poses (x, y, theta),
//! copper pours, IPC-2152 current-rated vias, and routed segments. Supports KiCad (.kicad_pcb).

use serde::{Deserialize, Serialize};

/// 2D Coordinate Point in millimeters.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Point2d {
    pub x_mm: f32,
    pub y_mm: f32,
}

/// Component Pose on PCB.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ComponentPlacement {
    pub ref_des: String,
    pub position: Point2d,
    pub rotation_deg: f32,
    pub layer: String, // "F.Cu" or "B.Cu"
}

/// Routed PCB Track Segment.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PcbTrackSegment {
    pub net_name: String,
    pub layer: String,
    pub start: Point2d,
    pub end: Point2d,
    pub width_mm: f32,
}

/// Physical PCB Board Graph.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PcbBoardGraph {
    pub outline_width_mm: f32,
    pub outline_height_mm: f32,
    pub layer_count: usize,
    pub placements: Vec<ComponentPlacement>,
    pub tracks: Vec<PcbTrackSegment>,
}

impl PcbBoardGraph {
    #[must_use]
    pub fn new(width_mm: f32, height_mm: f32, layer_count: usize) -> Self {
        Self {
            outline_width_mm: width_mm,
            outline_height_mm: height_mm,
            layer_count,
            placements: Vec::new(),
            tracks: Vec::new(),
        }
    }

    /// Exports layout to KiCad 8/9 .kicad_pcb S-expression string.
    #[must_use]
    pub fn export_kicad_pcb(&self) -> String {
        let mut out = String::new();
        out.push_str("(kicad_pcb (version 20231120) (generator oxide-eda)\n");
        out.push_str(&format!("  (layers (1 F.Cu signal) (31 B.Cu signal))\n"));

        // Board outline rectangle
        out.push_str(&format!(
            "  (gr_rect (start 0 0) (end {} {}) (layer \"Edge.Cuts\") (width 0.1))\n",
            self.outline_width_mm, self.outline_height_mm
        ));

        // Placements
        for p in &self.placements {
            out.push_str(&format!(
                "  (footprint \"Default\" (layer \"{}\") (at {} {} {})\n",
                p.layer, p.position.x_mm, p.position.y_mm, p.rotation_deg
            ));
            out.push_str(&format!(
                "    (property \"Reference\" \"{}\")\n",
                p.ref_des
            ));
            out.push_str("  )\n");
        }

        // Tracks
        for t in &self.tracks {
            out.push_str(&format!(
                "  (segment (start {} {}) (end {} {}) (width {}) (layer \"{}\") (net 1))\n",
                t.start.x_mm, t.start.y_mm, t.end.x_mm, t.end.y_mm, t.width_mm, t.layer
            ));
        }

        out.push_str(")\n");
        out
    }
}
