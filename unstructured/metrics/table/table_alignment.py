import difflib
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from unstructured_inference.models.eval import compare_contents_as_df


class TableAlignment:
    def __init__(self, cutoff: float = 0.8):
        self.cutoff = cutoff

    @staticmethod
    def get_content_in_tables(table_data: List[List[Dict[str, Any]]]) -> List[str]:
        # Replace below docstring with google-style docstring
        """Extracts and concatenates the content of cells from each table in a list of tables.

        Args:
          table_data: A list of tables, each table being a list of cell data dictionaries.

        Returns:
          List of strings where each string represents the concatenated content of one table.
        """
        return [" ".join([d["content"] for d in td if "content" in d]) for td in table_data]

    @staticmethod
    def get_table_level_alignment(
        predicted_table_data: List[List[Dict[str, Any]]],
        ground_truth_table_data: List[List[Dict[str, Any]]],
    ) -> List[int]:
        """Compares predicted table data with ground truth data to find the best
        matching table index for each predicted table.

        Args:
          predicted_table_data: A list of predicted tables.
          ground_truth_table_data: A list of ground truth tables.

        Returns:
          A list of indices indicating the best match in the ground truth for
          each predicted table.

        """
        ground_truth_texts = TableAlignment.get_content_in_tables(ground_truth_table_data)
        matched_indices = []
        for td in predicted_table_data:
            reference = TableAlignment.get_content_in_tables([td])[0]
            matches = difflib.get_close_matches(reference, ground_truth_texts, cutoff=0.1, n=1)
            matched_indices.append(ground_truth_texts.index(matches[0]) if matches else -1)
        return matched_indices

    @staticmethod
    def _zip_to_dataframe(table_data: List[Dict[str, Any]]) -> pd.DataFrame:
        df = pd.DataFrame(table_data, columns=["row_index", "col_index", "content"])
        df = df.set_index("row_index")
        df["col_index"] = df["col_index"].astype(str)
        return df

    @staticmethod
    def _find_unused_position(
        positions: List[int], used_indices: set
    ) -> tuple[int, set]:
        """Find the first unused position in the positions list.

        Args:
            positions: List of candidate position indices.
            used_indices: Set of already used indices.

        Returns:
            Tuple of (matched_idx, updated_used_indices).
            If all positions are used, resets used_indices and returns the first position.
        """
        for pos in positions:
            if pos not in used_indices:
                used_indices.add(pos)
                return pos, used_indices

        # If all indices are used, reset used_indices and use the first index
        used_indices.clear()
        used_indices.add(positions[0])
        return positions[0], used_indices

    @staticmethod
    def get_element_level_alignment(
        predicted_table_data: List[List[Dict[str, Any]]],
        ground_truth_table_data: List[List[Dict[str, Any]]],
        matched_indices: List[int],
        cutoff: float = 0.8,
    ) -> Dict[str, float]:
        """Aligns elements of the predicted tables with the ground truth tables at the cell level.

        Args:
          predicted_table_data: A list of predicted tables.
          ground_truth_table_data: A list of ground truth tables.
          matched_indices: Indices of the best matching ground truth table for each predicted table.
          cutoff: The cutoff value for the close matches.

        Returns:
          A dictionary with column and row alignment accuracies.

        """
        content_diff_cols = []
        content_diff_rows = []
        col_index_acc = []
        row_index_acc = []

        for idx, td in zip(matched_indices, predicted_table_data):
            if idx == -1:
                content_diff_cols.append(0)
                content_diff_rows.append(0)
                col_index_acc.append(0)
                row_index_acc.append(0)
                continue
            ground_truth_td = ground_truth_table_data[idx]

            # Get row and col content accuracy
            predict_table_df = TableAlignment._zip_to_dataframe(td)
            ground_truth_table_df = TableAlignment._zip_to_dataframe(ground_truth_td)

            table_content_diff = compare_contents_as_df(
                ground_truth_table_df.fillna(""),
                predict_table_df.fillna(""),
            )
            content_diff_cols.append(table_content_diff["by_col_token_ratio"])
            content_diff_rows.append(table_content_diff["by_row_token_ratio"])

            aligned_element_col_count = 0
            aligned_element_row_count = 0
            total_element_count = 0
            # Get row and col index accuracy
            ground_truth_td_contents_list = [gtd["content"].lower() for gtd in ground_truth_td]
            ground_map: Dict[str, List[int]] = {}
            unique_ground_list: List[str] = []
            for i, s in enumerate(ground_truth_td_contents_list):
                if s in ground_map:
                    ground_map[s].append(i)
                else:
                    ground_map[s] = [i]
                    unique_ground_list.append(s)

            used_indices = set()
            for td_ele in td:
                content = td_ele["content"].lower()
                row_index = td_ele["row_index"]
                col_idx = td_ele["col_index"]

                matched_idx = -1

                # Prefer exact matches (fast path) before fuzzy matching
                positions = ground_map.get(content)
                if positions is not None:
                    matched_idx, used_indices = TableAlignment._find_unused_position(
                        positions, used_indices
                    )
                else:
                    # Fallback to fuzzy matching on unique ground list (much smaller than full list)
                    matches = difflib.get_close_matches(
                        content,
                        unique_ground_list,
                        cutoff=cutoff,
                        n=1,
                    )
                    if matches:
                        match_str = matches[0]
                        positions = ground_map[match_str]
                        matched_idx, used_indices = TableAlignment._find_unused_position(
                            positions, used_indices
                        )

                if matched_idx >= 0:
                    gt_row_index = ground_truth_td[matched_idx]["row_index"]
                    gt_col_index = ground_truth_td[matched_idx]["col_index"]
                    # Update alignment counters immediately
                    if row_index == gt_row_index:
                        aligned_element_row_count += 1
                    if col_idx == gt_col_index:
                        aligned_element_col_count += 1
                    total_element_count += 1

            table_col_index_acc = 0
            table_row_index_acc = 0
            if total_element_count > 0:
                table_col_index_acc = round(aligned_element_col_count / total_element_count, 2)
                table_row_index_acc = round(aligned_element_row_count / total_element_count, 2)

            col_index_acc.append(table_col_index_acc)
            row_index_acc.append(table_row_index_acc)

        matched_set = set(matched_indices)
        not_found_gt_table_indexes = [
            id for id in range(len(ground_truth_table_data)) if id not in matched_set
        ]
        for _ in not_found_gt_table_indexes:
            content_diff_cols.append(0)
            content_diff_rows.append(0)
            col_index_acc.append(0)
            row_index_acc.append(0)

        return {
            "col_index_acc": round(np.mean(col_index_acc), 2),
            "row_index_acc": round(np.mean(row_index_acc), 2),
            "col_content_acc": round(np.mean(content_diff_cols) / 100.0, 2),
            "row_content_acc": round(np.mean(content_diff_rows) / 100.0, 2),
        }
