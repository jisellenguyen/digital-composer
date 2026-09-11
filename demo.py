
# cd "/Users/jisellenguyen/Documents/CS35/final project"


from snippet_generator import SnippetGenerator

gen = SnippetGenerator(model_path="output/model.pkl")

gen.generate_snippet(key="C", mode="minor", tempo=50, variability=0.2, 
                     voices=1, dynamics="loud", note_density=1.8, output_path="output/demo1.mid")

gen.generate_snippet(key="G", mode="major", tempo=90, variability=0.5, 
                     voices=2, dynamics="loud", note_density=1.8, output_path="output/demo2.mid")

gen.generate_snippet(key="F", mode="major", tempo=130, variability=0.8, 
                     voices=3, dynamics="loud", note_density=1.8, output_path="output/demo3.mid")