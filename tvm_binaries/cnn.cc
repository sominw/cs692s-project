
#include "ops.h"
#include "squeezenet.h"
#include "resnet.h"
#include <cstring> 

Graph* optimize_graph(Graph *graph, Model *model, float beta, int budget)
{
  std::vector<GraphXfer*> xfers;
  xfers.push_back(create_fuse_conv_batch_xfer(model));
  xfers.push_back(create_fuse_mm_acti_xfer(model));
  xfers.push_back(create_fuse_conv_relu_xfer(model));
  xfers.push_back(create_merge_mm_xfer(model));
  xfers.push_back(create_merge_conv_xfer(model));
  xfers.push_back(create_exclusive_concat_xfer(model));
  xfers.push_back(create_resnet_merge_xfer(model));

  std::priority_queue<Graph*, std::vector<Graph*>, GraphCompare> candidates;
  std::set<size_t> hashmap;
  candidates.push(graph);
  hashmap.insert(graph->hash());
  Graph *bestGraph = graph;
  float bestCost = graph->total_cost();
  printf("Baseline Graph:\n    End-to-end runtime = %.4lfms\n", graph->run(model));
  graph->print_costs();

  int counter = 0;
  bool firstGraph = true;
  std::map<Edge, int, EdgeCompare> edgeWeights;
  while (!candidates.empty()) {
    Graph *subGraph = candidates.top();
    candidates.pop();
    if (subGraph->total_cost() < bestCost) {
      delete bestGraph;
      bestCost = subGraph->total_cost();
      bestGraph = subGraph;
    }
    if (subGraph->total_cost() > beta * bestCost) {
      delete subGraph;
      continue;
    }
    if (counter > budget) {
      break;
    }
#ifdef VERBOSE
    if (counter % 100 == 0)
      printf("[%d] cost = %.4lf bestCost = %.4lf candidates.size() = %zu\n", counter, subGraph->total_cost(), bestCost, candidates.size());
#endif
    counter ++;
    for (int i = 0; i < xfers.size(); i++)
      xfers[i]->run(0, subGraph, candidates, hashmap, bestCost * beta, edgeWeights, firstGraph);
    firstGraph = false;
    if (bestGraph != subGraph) {
      delete subGraph;
    }
  }
  printf("Optimized Graph:\n    End-to-end runtime = %.4lfms\n", bestGraph->run(model));
  bestGraph->print_costs();

  GraphXfer::print_edge_weights(edgeWeights);
  return bestGraph;
}

enum DNNModel {
  None,
  SqueezeNet,
  Resnet34,
  Resnet50,
  Resnet18
};

DNNModel name_to_model(std::string name)
{
  if (name == "squeezenet") return SqueezeNet;
  if (name == "resnet34") return Resnet34;
  if (name == "resnet50") return Resnet50;
  if (name == "resnet18") return Resnet18;
  assert(false);
}

void parse_args(bool &optimize,
                bool &export_graph,
                float &beta,
                int &budget,
                std::string &export_file_name,
                DNNModel &dnnModel,
                int argc,
                char **argv)
{
  std::string dnnName;
  for (int i = 1; i < argc; i++)
  {
    if (!strcmp(argv[i],"--noopt")) {
      optimize = false;
      continue;
    }
    if (!strcmp(argv[i], "--budget")) {
      budget = std::atoi(argv[++i]);
      continue;
    }
    if (!strcmp(argv[i],"--export")) {
      export_graph = true;
      export_file_name = argv[++i];
      continue;
    }
    if (!strcmp(argv[i],"--beta")) {
      beta = std::atof(argv[++i]);
      continue;
    }
    if (!strcmp(argv[i],"--dnn")) {
      dnnName = std::string(argv[++i]);
      continue;
    }

    fprintf(stderr, "Found unknown option!!\n");
    assert(0);
  }
  dnnModel = name_to_model(dnnName);
}

int main(int argc, char **argv)
{
  bool optimize = true;
  bool export_graph = false;
  int budget = 300; 
  float beta = 1.01;
  DNNModel dnn = None;
  std::string export_file_name;
  parse_args(optimize, export_graph, beta, budget, export_file_name, dnn, argc, argv);
  assert(dnn != None);
  printf("DnnModel(%d) beta(%.4lf)\n", dnn, beta);

  Model* model = new Model(false);
  Graph* graph = NULL;
  switch (dnn) {
    case SqueezeNet:
      graph = SqueezeNetComplex(model);
      break;
    case Inception:
      graph = InceptionV3(model);
      break;
    case Resnet34:
      graph = ResNet34(model);
      break;
    case Resnet50:
      graph = ResNet50(model);
      break;
    case DenseNet:
      graph = DenseNet121(model);
      break;
    case RNNTC:
      graph = RNNTC_SRU(model);
      break;
    default:
      assert(false);
  }
#ifdef TRT
  void runGraphTRT(Graph *graph);
#endif
  if (optimize && dnn == RNNTC && false){
    printf("Baseline Graph:\n");
    printf("    End-to-end runtime = %.4lf\n", graph->run(model));
    graph->print_costs();
    graph = RNNTC_OPT(model);
    printf("Optimized Graph:\n"); 
    printf("    End-to-end runtime = %.4lf\n", graph->run(model));
    graph->print_costs();
  } else if (optimize) {
    graph = optimize_graph(graph, model, beta, budget);
  }
  if (export_graph)
  {
    graph->export_to_file(export_file_name);
  }
#ifdef TRT
  runGraphTRT(graph);
#endif
  return 0;
}

int example(Model* model)
{
  Graph *graph = new Graph(model);
  Tensor input;
  input.numDim = 4;
  input.dim[0] = BATCH_SIZE;
  input.dim[1] = 384;
  input.dim[2] = 8;
  input.dim[3] = 8;
  input.op.guid = 0;
  input.op.ptr = NULL;
  input.idx = 0;
  Tensor t = graph->conv2d(input, 384, 3, 3, 1, 1, 1, 1, true);
  Tensor t1 = graph->conv2d(t, 384, 3, 3, 1, 1, 1, 1, true);
  Tensor t2 = graph->conv2d(t, 384, 3, 3, 1, 1, 1, 1, true);
}
