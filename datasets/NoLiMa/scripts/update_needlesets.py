import json
import glob
import os

input_filepaths = glob.glob("needlesets/needle_set*.json")
example_format_1 = """
{
  "lines": [25, 412],
  "answer": "John",
}"""

example_format_2 = """
{
  "lines": ["John is suffering from malaria"],
  "answer": "John",
}"""

line_number_demo = """let us consider the following context having 4 sentences:\n<context>\n\n\nSentence1\nSentence2\n\nSentence3\n\n\nSentence4\n</context>\nIn the above context, line numbers of each sentence are as follows: Sentence1=[3],Sentence2=[4],Sentence3=[6],Sentence4=[9]"""

def task1():
  output_filepaths = [f"task-1_{fpath}" for fpath in input_filepaths]
  for inp_fpath, out_fpath in zip(input_filepaths, output_filepaths):
      os.makedirs(out_fpath.split("/")[0], exist_ok=True)
      with open(out_fpath, "w") as wf:
          with open(inp_fpath, "r") as in_rf:
              data = json.load(in_rf)
              # print(in_data[0])

              data_updated = []
              for row in data:
                  # if row["reasoning_type"] == "commonsense_knowledge":
                  row["task_template"] = f"You will answer a question based on the following book snippet:\n\n<context>{{haystack}}</context>\n\nUse the information provided in the book snippet to answer the question. If the question is not answerable from the book snippet, respond with \"NA\" as the answer. Your answer should be short and based on explicitly stated facts from the mentioned book snippet only. Return only the final answer and all lines (considering \"\n\" as line breaks) the answer is based on using **zero-based indexing** (i.e., line numbering starts at 0) in json format. For example:\n```json{example_format_1}\n```\n\nQuestion: {{question}}".replace("json\n{\n", "json\n{{\n").replace("\n```", "}\n```").replace(":}", ":")
                  data_updated.append(row)

          json.dump(data_updated, wf, ensure_ascii=False, indent=2)



def task2():
  output_filepaths = [f"task-2_{fpath}" for fpath in input_filepaths]
  system_prompt = "Your job is to answer the question entirely from the context and provide a reference. Your answer should cite all lines the answer is based on."
  for inp_fpath, out_fpath in zip(input_filepaths, output_filepaths):
      os.makedirs(out_fpath.split("/")[0], exist_ok=True)
      with open(out_fpath, "w") as wf:
          with open(inp_fpath, "r") as in_rf:
              data = json.load(in_rf)
              # print(in_data[0])

              data_updated = []
              for row in data:
                  # if row["reasoning_type"] == "commonsense_knowledge":
                  row["system_prompt"] = system_prompt
                  row["task_template"] = f"<context>{{haystack}}</context>\n\nAnswer the question based on information only from the context. If the question is not answerable from the context, answer NA. Your response should comprise only the answer and all lines the answer is based on in json format. For example:\n```json{example_format_1}\n```\n\nQuestion: {{question}}".replace("json\n{\n", "json\n{{\n").replace("\n```", "}\n```").replace(":}", ":")
                  data_updated.append(row)

          json.dump(data_updated, wf, ensure_ascii=False, indent=2)


def task3():
  output_filepaths = [f"task-3_{fpath}" for fpath in input_filepaths]
  system_prompt = "Your job is to answer the question entirely from the context and provide a reference. Your answer should cite all lines the answer is based on."
  for inp_fpath, out_fpath in zip(input_filepaths, output_filepaths):
      os.makedirs(out_fpath.split("/")[0], exist_ok=True)
      with open(out_fpath, "w") as wf:
          with open(inp_fpath, "r") as in_rf:
              data = json.load(in_rf)
              # print(in_data[0])

              data_updated = []
              for row in data:
                  # if row["reasoning_type"] == "commonsense_knowledge":
                  row["system_prompt"] = system_prompt
                  row["task_template"] = f"<context>{{haystack}}</context>\n\nAnswer the question based on information only from the context. If the question is not answerable from the context, answer NA. Your response should comprise only the answer and all lines (considering \"\n\" as line breaks) the answer is based on using **zero-based indexing** (i.e., line numbering starts at 0) in json format. For example:\n```json{example_format_1}\n```\n\nQuestion: {{question}}".replace("json\n{\n", "json\n{{\n").replace("\n```", "}\n```").replace(":}", ":")
                  data_updated.append(row)

          json.dump(data_updated, wf, ensure_ascii=False, indent=2)

def task3p1():
  output_filepaths = [f"task-3p1_{fpath}" for fpath in input_filepaths]
  system_prompt = "Your job is to answer the question entirely from the context and provide a reference. Your answer should cite all lines the answer is based on."
  for inp_fpath, out_fpath in zip(input_filepaths, output_filepaths):
      os.makedirs(out_fpath.split("/")[0], exist_ok=True)
      with open(out_fpath, "w") as wf:
          with open(inp_fpath, "r") as in_rf:
              data = json.load(in_rf)
              # print(in_data[0])

              data_updated = []
              for row in data:
                  # if row["reasoning_type"] == "commonsense_knowledge":
                  row["system_prompt"] = system_prompt
                  row["task_template"] = f"<context>{{haystack}}</context>\n\nAnswer the question based on information only from the context. If the question is not answerable from the context, answer NA. Your response should comprise only the answer and all lines (considering \"\n\" as line breaks) the answer is based on using **zero-based indexing** (i.e., line numbering starts at 0) in json format. Provide only the line numbers, not the text in the line. For example:\n```json{example_format_1}\n```\n\nQuestion: {{question}}".replace("json\n{\n", "json\n{{\n").replace("\n```", "}\n```").replace(":}", ":")
                  data_updated.append(row)

          json.dump(data_updated, wf, ensure_ascii=False, indent=2)

def task3p2():
  output_filepaths = [f"task-3p2_{fpath}" for fpath in input_filepaths]
  system_prompt = "Your job is to answer the question entirely from the context and provide a reference. Your answer should cite all lines the answer is based on."
  for inp_fpath, out_fpath in zip(input_filepaths, output_filepaths):
      os.makedirs(out_fpath.split("/")[0], exist_ok=True)
      with open(out_fpath, "w") as wf:
          with open(inp_fpath, "r") as in_rf:
              data = json.load(in_rf)
              # print(in_data[0])

              data_updated = []
              for row in data:
                  # if row["reasoning_type"] == "commonsense_knowledge":
                  row["system_prompt"] = system_prompt
                  row["task_template"] = f"<context>{{haystack}}</context>\n\nAnswer the question based on information only from the context. If the question is not answerable from the context, answer NA. Your response should comprise only the answer and all lines (i.e., evidence) the answer is based on in json format. Provide the evidence text and not the line numbers. For example:\n```json{example_format_2}\n```\n\nQuestion: {{question}}".replace("json\n{\n", "json\n{{\n").replace("\n```", "}\n```").replace(":}", ":")
                  data_updated.append(row)

          json.dump(data_updated, wf, ensure_ascii=False, indent=2)

def task4():
  output_filepaths = [f"task-4_{fpath}" for fpath in input_filepaths]
  system_prompt = "Your job is to answer the question entirely from the context and provide a reference. Your answer should cite all lines the answer is based on."
  for inp_fpath, out_fpath in zip(input_filepaths, output_filepaths):
      os.makedirs(out_fpath.split("/")[0], exist_ok=True)
      with open(out_fpath, "w") as wf:
          with open(inp_fpath, "r") as in_rf:
              data = json.load(in_rf)
              # print(in_data[0])

              data_updated = []
              for row in data:
                  # if row["reasoning_type"] == "commonsense_knowledge":
                  row["system_prompt"] = system_prompt
                  row["task_template"] = f"<context>{{haystack}}</context>\n\nAnswer the question based on information only from the context. If the question is not answerable from the context, answer NA. Your response should comprise only the answer and all lines (considering \"\n\" as line breaks) the answer is based on using **zero-based indexing** (i.e., line numbering starts at 0, e.g., {line_number_demo}) in json format. For example:\n```json{example_format_1}\n```\n\nQuestion: {{question}}".replace("json\n{\n", "json\n{{\n").replace("\n```", "}\n```").replace(":}", ":")
                  data_updated.append(row)

          json.dump(data_updated, wf, ensure_ascii=False, indent=2)

def task5():
  output_filepaths = [f"task-5_{fpath}" for fpath in input_filepaths]
  system_prompt = "Your job is to answer the question entirely from the context and provide a reference. Your answer should cite all lines the answer is based on."
  for inp_fpath, out_fpath in zip(input_filepaths, output_filepaths):
      os.makedirs(out_fpath.split("/")[0], exist_ok=True)
      with open(out_fpath, "w") as wf:
          with open(inp_fpath, "r") as in_rf:
              data = json.load(in_rf)
              # print(in_data[0])

              data_updated = []
              for row in data:
                  # if row["reasoning_type"] == "commonsense_knowledge":
                  row["system_prompt"] = system_prompt
                  row["task_template"] = f"<context>{{haystack}}</context>\n\nAnswer the question based on information only from the context. If the question is not answerable from the context, answer NA. Your response should comprise only the answer and all lines the answer is based on in json format. For example:\n```json{example_format_1}\n```\n\nQuestion: {{question}}".replace("json\n{\n", "json\n{{\n").replace("\n```", "}\n```").replace(":}", ":")
                  data_updated.append(row)

          json.dump(data_updated, wf, ensure_ascii=False, indent=2)

def task5p1():
  output_filepaths = [f"task-5p1_{fpath}" for fpath in input_filepaths]
  system_prompt = "Your job is to answer the question entirely from the context and provide a reference. Your answer should cite all lines the answer is based on."
  for inp_fpath, out_fpath in zip(input_filepaths, output_filepaths):
      os.makedirs(out_fpath.split("/")[0], exist_ok=True)
      with open(out_fpath, "w") as wf:
          with open(inp_fpath, "r") as in_rf:
              data = json.load(in_rf)
              # print(in_data[0])

              data_updated = []
              for row in data:
                  # if row["reasoning_type"] == "commonsense_knowledge":
                  row["system_prompt"] = system_prompt
                  row["task_template"] = f"<context>{{haystack}}</context>\n\nAnswer the question based on information only from the context. If the question is not answerable from the context, answer NA. Your response should comprise only the answer (just the character name) and all lines the answer is based on in json format. For example:\n```json{example_format_1}\n```\n\nQuestion: {{question}}".replace("json\n{\n", "json\n{{\n").replace("\n```", "}\n```").replace(":}", ":")
                  data_updated.append(row)

          json.dump(data_updated, wf, ensure_ascii=False, indent=2)

def main():
  task1()
  task2()
  task3()
  task3p1()
  task3p2()
  task4()
  task5()
  task5p1()

if __name__ == "__main__":
  main()

