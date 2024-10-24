import os
import json
from tqdm import tqdm
from feverous.database.feverous_db import FeverousDB
from feverous.utils.wiki_page import WikiPage

from feverous.database.feverous_db import FeverousDB
from feverous.utils.wiki_page import WikiPage
from feverous.utils.util import get_evidence_by_table, get_evidence_text_by_id, get_evidence_by_page
from feverous.utils.annotation_processor import AnnotationProcessor
from feverous.utils.wiki_table import Cell


class FEVEROUSINSERTER():

    def __init__(self, file_name):
        self.feverous_db = "data/feverous/feverous_wikiv1.db"
        self.feverous_annotations = "data/feverous/feverous_dev_challenges.jsonl"
        # self.feverous_annotations = "data/feverous/feverous_train_challenges.jsonl"

        self.processed_data_path = os.path.join("data", "feverous", "{}.jsonl".format(file_name))

        self.all_annotations = self.format_and_fill_feverous()



    def format_and_fill_feverous(self):
        db =  FeverousDB(self.feverous_db)

        annotation_processor = AnnotationProcessor(self.feverous_annotations) #limit=1000
        all_annotations = []
        for anno in tqdm(annotation_processor):
            content = {}
            content["id"] = anno.id
            content["claim"] = anno.claim
            content["verdict"] = anno.verdict
            content["challenge"] = anno.annotation_json["challenge"]
            # content["annotator_operations"] = anno.operations
            content["sentence_evidence"] = []
            content["table_evidence"] = []
            content["list_evidence"] = []
            content["selected_cells"] = []
            content["selected_list_items"] = []

            all_pages = {}
            table_ids = set([])
            list_ids = set([])

            unique_pages = list(dict.fromkeys([x.split("_")[0] for x in anno.evidence[0]])) # Only display the first evidence set for now.

            if any(["sentence" in ev for ev in anno.flat_evidence]): # If not all evidence set are exclusively tabular (or lists), skip
                continue


            for page in unique_pages:
                page_json = db.get_doc_json(page)
                all_pages[page] = (WikiPage(page, page_json), page_json)
            for evidence in anno.evidence[0]: # Ignore multiple evidence sets for now. Only 10\% of claims have more than 1.
                page = evidence.split("_")[0]
                ev_id = "_".join(evidence.split("_")[1:])
                evidence_type = anno.get_evidence_type()[0].name
                content["evidence_type"] = evidence_type
                if "sentence" in ev_id:
                    sentence_content = all_pages[page][0].page_items[ev_id]
                    sentence_id = page + "_" + ev_id
                    content["sentence_evidence"].append({"content" : str(sentence_content), "id": sentence_id, "context": [str(x) for x in all_pages[page][0].get_context(ev_id)]})
                elif "cell" in ev_id:
                    table = all_pages[page][0].get_table_from_cell_id(ev_id)
                    table_name_w_title = page + "_" + table.name
                    if  table_name_w_title not in table_ids:
                        table_content = all_pages[page][1][table.name]["table"] # TODO: This is a workaround to get table content, since table.table is erroneous, https://github.com/Raldir/FEVEROUS/issues/28
                        context = all_pages[page][0]._get_section_context(table.name)
                        table_context = [str(x) for x in context]
                        table_ids.add(table_name_w_title)
                        for row in table_content:
                            for col in row:
                                cell_context = all_pages[page][0].get_context(col["id"])
                                # print([type(x) for x in cell_context])
                                cell_context = [x for x in cell_context if isinstance(x, Cell)]
                                col["context"] = [str(x).replace("[H]","").strip() for x in cell_context]
                                # print(col)
                        content["table_evidence"].append({"content": table_content, "id": table_name_w_title, "context": table_context})
                    content["selected_cells"].append(page + "_" + ev_id)
                elif "item" in ev_id:
                    for wiki_list in all_pages[page][0].get_lists():
                        list_name_w_title = page + "_" + wiki_list.name
                        if ev_id in wiki_list.list_items:
                            if list_name_w_title not in list_ids:
                                list_ids.add(list_name_w_title)
                                list_content = wiki_list.list_items
                                context = all_pages[page][0]._get_section_context(wiki_list.name)
                                list_context = [str(x) for x in context]
                                content["list_evidence"].append({"content": list_content, "id": list_name_w_title, "context": list_context})
                    content["selected_list_items"].append(page + "_" + ev_id)
            all_annotations.append(content)

        print("Writing to {} ...".format(self.processed_data_path))
        with open(self.processed_data_path, "w", encoding="utf-8") as f_out:
            for content in all_annotations:
                f_out.write("{}\n".format(json.dumps(content)))
        return all_annotations


if __name__ == "__main__":
    FEVEROUSINSERTER("feverous_dev_filled_tables.jsonl")