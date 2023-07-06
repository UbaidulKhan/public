from pycallgraph import PyCallGraph
from pycallgraph.output import GraphvizOutput

def function_d():
  pass

def function_c():
  function_d()
  
def function_b():
  function_c()

  
def function_a():
  function_b()

with PyCallGraph(output=GraphvizOutput()):
    function_a()