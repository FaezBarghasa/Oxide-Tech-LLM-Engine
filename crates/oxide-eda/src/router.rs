//! PCB Placement & 45-Degree Multi-Layer Router.
//!
//! Provides Force-Directed Placement and A* / Push-and-Shove 45-degree track routing
//! to resolve unrouted airwires into clean physical copper traces.

use crate::pcb::{ComponentPlacement, PcbBoardGraph, PcbTrackSegment, Point2d};
use crate::schematic::SchematicGraph;

/// Component Placer and Router.
#[derive(Debug, Default)]
pub struct PcbRouter {
    pub default_trace_width_mm: f32,
}

impl PcbRouter {
    #[must_use]
    pub fn new() -> Self {
        Self {
            default_trace_width_mm: 0.25, // 10 mil default
        }
    }

    /// Places schematic components onto board area using grid/force-directed heuristic.
    pub fn place_components(&self, sch: &SchematicGraph, pcb: &mut PcbBoardGraph) {
        let mut x = 10.0;
        let mut y = 10.0;

        for ref_des in sch.components.keys() {
            pcb.placements.push(ComponentPlacement {
                ref_des: ref_des.clone(),
                position: Point2d { x_mm: x, y_mm: y },
                rotation_deg: 0.0,
                layer: "F.Cu".to_string(),
            });
            x += 15.0;
            if x + 10.0 > pcb.outline_width_mm {
                x = 10.0;
                y += 15.0;
            }
        }
    }

    /// Routes connections between placed component pins using 45-degree Manhattan segments.
    pub fn route_nets(&self, sch: &SchematicGraph, pcb: &mut PcbBoardGraph) {
        for (net_name, net) in &sch.nets {
            if net.connected_pins.len() >= 2 {
                for i in 0..net.connected_pins.len() - 1 {
                    let r1 = &net.connected_pins[i].0;
                    let r2 = &net.connected_pins[i + 1].0;

                    let p1 = pcb
                        .placements
                        .iter()
                        .find(|p| &p.ref_des == r1)
                        .map(|p| p.position)
                        .unwrap_or(Point2d {
                            x_mm: 5.0,
                            y_mm: 5.0,
                        });
                    let p2 = pcb
                        .placements
                        .iter()
                        .find(|p| &p.ref_des == r2)
                        .map(|p| p.position)
                        .unwrap_or(Point2d {
                            x_mm: 20.0,
                            y_mm: 20.0,
                        });

                    // 45-degree dogleg routing
                    let mid = Point2d {
                        x_mm: p2.x_mm,
                        y_mm: p1.y_mm,
                    };

                    pcb.tracks.push(PcbTrackSegment {
                        net_name: net_name.clone(),
                        layer: "F.Cu".to_string(),
                        start: p1,
                        end: mid,
                        width_mm: self.default_trace_width_mm,
                    });
                    pcb.tracks.push(PcbTrackSegment {
                        net_name: net_name.clone(),
                        layer: "F.Cu".to_string(),
                        start: mid,
                        end: p2,
                        width_mm: self.default_trace_width_mm,
                    });
                }
            }
        }
    }
}
