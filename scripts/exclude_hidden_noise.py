"""Rebuild a bundle so every presentation-hidden noise vertex is excluded."""
import argparse
import json
from soyrootbio.noise_bundle import exclude_hidden_noise

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source')
    parser.add_argument('target')
    args = parser.parse_args()
    report = exclude_hidden_noise(args.source, args.target)
    print(json.dumps({k: report[k] for k in ('excluded_vertex_count', 'hidden_assigned_vertex_count',
        'removed_noise_only_root_ids', 'unmeasured_topology_placeholder_ids', 'fitting_component_changed_root_ids', 'system_summary')}, indent=2))
