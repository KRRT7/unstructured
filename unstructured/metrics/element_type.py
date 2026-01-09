from __future__ import annotations

import json

from typing_extensions import TypeAlias

FrequencyDict: TypeAlias = "dict[tuple[str, int | None], int]"
"""Like:
    {
        ("ListItem", 0): 2,
        ("NarrativeText", None): 2,
        ("Title", 0): 5,
        ("UncategorizedText", None): 6,
    }
"""


def get_element_type_frequency(
    elements: str,
) -> FrequencyDict:
    """
    Calculate the frequency of Element Types from a list of elements.

    Args:
        elements (str): String-formatted json of all elements (as a result of elements_to_json).
    Returns:
        Element type and its frequency in dictionary format.
    """
    frequency: dict[tuple[str, int | None], int] = {}
    if len(elements) == 0:
        return frequency
    for element in json.loads(elements):
        type = element.get("type")
        category_depth = element["metadata"].get("category_depth")
        key = (type, category_depth)
        if key not in frequency:
            frequency[key] = 1
        else:
            frequency[key] += 1
    return frequency


def calculate_element_type_percent_match(
    output: FrequencyDict,
    source: FrequencyDict,
    category_depth_weight: float = 0.5,
) -> float:
    """Calculate the percent match between two frequency dictionary.

    Intended to use with `get_element_type_frequency` function. The function counts the absolute
    exact match (type and depth), and counts the weighted match (correct type but different depth),
    then normalized with source's total elements.
    """
    if len(output) == 0 or len(source) == 0:
        return 0.0

    output_copy = output.copy()
    source_copy = source.copy()
    total_source_element_count = 0
    total_match_element_count = 0

    unmatched_depth_output: dict[str, int] = {}

    # First pass: perform direct matches and accumulate unmatched by type
    for k, v in output_copy.items():
        output_count = v
        source_count = source_copy.get(k, 0)
        if source_count:
            match_count = min(output_count, source_count)
            total_match_element_count += match_count
            total_source_element_count += match_count
            output_count -= match_count
            source_count -= match_count
            if source_count:
                source_copy[k] = source_count
            else:
                # Remove for later for speed (avoid accumulating in _convert step)
                source_copy.pop(k)
        # prepare unmatched by type if any left
        if output_count:
            elem_type = k[0]
            unmatched_depth_output[elem_type] = (
                unmatched_depth_output.get(elem_type, 0) + output_count
            )

    # Fast inline _convert_to_frequency_without_depth for source leftovers only >0
    unmatched_depth_source: dict[str, int] = {}
    for (elem_type, _), v in source_copy.items():
        if v:
            unmatched_depth_source[elem_type] = unmatched_depth_source.get(elem_type, 0) + v

    # Second pass: weighted matches by type (depth-insensitive)
    for elem_type, v in unmatched_depth_source.items():
        total_source_element_count += v
        count_in_output = unmatched_depth_output.get(elem_type)
        if count_in_output:
            match_count = min(count_in_output, v)
            total_match_element_count += match_count * category_depth_weight

    return min(max(total_match_element_count / total_source_element_count, 0.0), 1.0)


def _convert_to_frequency_without_depth(d: FrequencyDict) -> dict[str, int]:
    """
    Takes in element frequency with depth of format (type, depth): value
    and converts to dictionary without depth of format type: value
    """
    res: dict[str, int] = {}
    for k, v in d.items():
        element_type = k[0]
        if element_type not in res:
            res[element_type] = v
        else:
            res[element_type] += v
    return res
