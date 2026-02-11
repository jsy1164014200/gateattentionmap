import numpy as np
from gurobipy import Model, GRB
import json

from langchain.tools import tool
from langchain.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langchain_qwq import ChatQwen
from langgraph.checkpoint.memory import InMemorySaver
from langchain.agents import create_agent


# OR agent: OR-LLM-agent 

# reasonging LLM as heuristic solver

# tool use: decision variables, obj, cons. LP





np.random.seed(0)

range_of_coef_min = -100
range_of_coef_max = 100 
range_of_y_min = -100
range_of_y_max = 100

x_dim = 10 
num_x_cons = 5 

c = np.random.randint(range_of_coef_min, range_of_coef_max, size=(x_dim,)) 
A = np.random.randint(range_of_coef_min, range_of_coef_max, size=(num_x_cons, x_dim)) 
b = np.random.randint(range_of_coef_min, range_of_coef_max, size=(num_x_cons,)) 
num_snr = 4 
y_dim = 20 
num_y_cons = 10
q = np.random.randint(range_of_coef_min, range_of_coef_max, size=(num_snr, y_dim,))
T = np.random.randint(range_of_coef_min, range_of_coef_max, size=(num_snr, num_y_cons, x_dim)) 
W = np.random.randint(range_of_coef_min, range_of_coef_max, size=(num_snr, num_y_cons, y_dim))
h = np.random.randint(range_of_coef_min, range_of_coef_max, size=(num_snr, num_y_cons,))


# tool input: decision variable, obj, cons, LP. 

# tool description (docstring) -> tool description
@tool
def solve_second_stage(x_val: list[int]) -> str:
    """求解两阶段优化问题的第二阶段问题，输入为第一阶段整数解x_val，输出为第二阶段问题的最优值和最优解的json字符串，其中second_stage_cost为第二阶段问题的最优值，snr_costs为每个不确定场景下第二阶段问题的最优值列表，optimal_y为每个不确定场景下第二阶段问题的最优解列表。"""
    np.array(x_val)
    total_cost = 0
    snr_cost = []
    optimal_y = []
    for s in range(num_snr):
        sub_m = Model(f"second_stage_snr_{s}")
        sub_m.setParam("OutputFlag", 0)
        y_s = sub_m.addMVar(y_dim, lb=range_of_y_min, ub=range_of_y_max, vtype=GRB.CONTINUOUS, name=f"y_{s}")
        sub_m.setObjective(q[s] @ y_s, GRB.MINIMIZE)
        sub_m.addConstr(T[s] @ x_val + W[s] @ y_s == h[s], name=f"snr_{s}_TxWy_eq_h")
        sub_m.optimize()
        if sub_m.status == GRB.OPTIMAL:
            total_cost += sub_m.objVal
            snr_cost.append(sub_m.objVal)
            optimal_y.append(y_s.X.tolist())
        else:
            print(f"Scenario {s} second stage problem not optimal, status code:", sub_m.status)
    rt_dict = {
        "second_stage_cost": total_cost / num_snr,
        "snr_costs": snr_cost,
        "optimal_y": optimal_y
    }
    return json.dumps(rt_dict)



system_message = SystemMessage("""你是一个运筹优化领域的专家，你非常擅长使用你的推理能力求解两阶段的优化问题。

具体来说，你擅长的两阶段问题的第一个阶段会涉及到整数变量，而第二个阶段则是一个线性规划问题。你会根据用户给出的问题描述和问题数据，先猜测出一个不错的第一阶段整数解(包含推理逻辑和理由)，然后在必要时调用工具来求解第二阶段的问题（工具会告诉你给定你猜测的一阶段解情况下第二阶段问题的最优值和最优解），最后你会根据这些信息来更新解。重复这个过程最多三次，直到给出你认为的近似最优解。
                               
你有如下一个工具可以使用：
- solve_second_stage: 这个工具会接受一个整数解作为输入，返回第二阶段问题在该整数解下的最优值和最优解的信息。 
""")



model = ChatQwen(
    base_url="https://1yvxf19722895.vicp.fun/v1",  
    api_key="xxx", 
    model="Qwen/Qwen3-8B",      
    # temperature=0, 
    # max_tokens=32768, 
    timeout=200,
    # max_retries=2, 
    # enable_thinking=True,
) 





agent = create_agent(
    model=model,
    system_prompt=system_message,
    tools=[solve_second_stage],  
   #  checkpointer=InMemorySaver(), 
)

if __name__ == "__main__":
    config = {"configurable": {"thread_id": "1"}}  
    input_message = HumanMessage("""请帮我求解一下这个两阶段随机规划问题对应的两阶段混合整数规划问题。
                                 
$$
min  & c^T x + E[ Q(x, xi) ] 

s.t. & x in {0,1}^{x_dim}.
$$

$$
Q(x, xi) = min  & q(xi)^T y 

s.t. & T(xi) x + W(xi) y = h(xi), 

& range_of_y_min <= y <= range_of_y_max.
$$
                                 
问题的数据如下：
- x_dim = 10 第一阶段变量x的维度 
- num_snr = 4 不确定场景数
- y_dim = 20 第二阶段变量y的维度
- range_of_y_min = -100
- range_of_y_max = 100
- num_y_cons = 10 第二阶段约束数
                                 
c: [ 72 -53  17  92 -33  95   3 -91 -79 -64] 
q: [[ -69   20  -99  -35   69  -43  -65    2   19  -89   74  -18   -9   28
    42   -1  -47   40   21   70]
 [ -16  -32  -94   96  -53   27   31    0   80  -22   43   48   86  -77
    41   17  -15  -52  -51  -31]
 [  69   63   92   -5   97   -6 -100   13   78  -64   62  -52   -7   31
    -2  -58   12   49   27 -100]
 [  38   14  -57   86   27  -77   87   30   21   -2  -38   63   23   95
   -18   74   48  -50   55  -86]]
T: [[[ -59  -42   93  -64  -90  -14  -57    4  -89  -98]
  [ -49  -20  -68   82   28  -62  -81   74  -58   15]
  [  84   88  -23  -70  -76   25  -98  -97   -6    7]
  [ -87   12  -60  -28  -81   -5  -28   54   94   80]
  [ -33  -39  -86   -4  -96   95   39  -14   21    9]
  [ -25   84  -84   52   57   49   10  -75   88   21]
  [  18   17   89  -17   61    4   60   21  -30  -69]
  [ -87  -29   84   52  -21  -59  -82  -60   82  -89]
  [  66   11   -7   29   18  -56   25  -76  -33  -97]
  [ -65   89   97  -57  -68  -89    4   38   82   65]]

 [[  25   56   11  -98  -73   51  -47  -49   74   48]
  [  81  -71  -33  -65  -61   37  -27  -59   51   31]
  [ -54   78    8  -97  -69  -91   38  -73   73   99]
  [  67  -39  -15   -3  -56  -66   62  -12  -67   33]
  [ -64 -100  -66   97   26   81  -20   90   36   89]
  [  29   12  -65   20   -9   68   16  -64   76  -75]
  [ -33    3  -65   14  -70  -71  -67   46  -83  -16]
  [ -98  -31    1   40  -56   17  -34   11   -9  -15]
  [  67  -61   50   58   45   98   99  -82   -8  -57]
  [ -17   77  -59   -7   74   49  -11  -27  -72    5]]

 [[  86   28  -37  -84    6   64   -6  -76   16   91]
  [  95  -49   36   84   -9   -7   23  -13   60   47]
  [ -28   99  -13  -87  -42  -19   20   16   83  -36]
  [  64  -75  -68   70  -86  -72  -80  -32  -78   22]
  [ -17   35  -39   41  -95 -100   36   81   39  -96]
  [  67   -8   73  -74  -26  -48   77  -49    5  -82]
  [  17  -66  -49   58   81  -42   71  -45  -82   73]
  [ -13   93  -30  -47  -52   -6  -41  -20   54   24]
  [  63  -42   77    6  -56  -87   21  -30  -62   67]
  [  36  -87   35  -78  -21  -92  -94   99   94  -40]]

 [[  44  -44   14   51  -76  -96    0  -51  -13  -70]
  [ -46   53  -80   -3    1   85   51   55  -71   61]
  [  15  -47   19   79  -14  -93    5   37   82   28]
  [ -17   20   64   48   17  -97   26  -58  -35  -80]
  [ -64  -32   12   75   38    4   -9  -57  -37   59]
  [  48   98  -91   88   -9   11   63  -17  -24  -82]
  [  13  -26   71   31   40  -42   29   13   28  -61]
  [ -76   86  -64   -1  -31   34  -97   21   68   88]
  [  61  -72  -32  -74    9   79   81   97   61   35]
  [  25   -6  -28  -16   35   95    8  -33    2  -16]]]
W: [[[ -29  -17 -100   33   -9    7   58  -93   49   36   71  -54 -100    4
     79  -62  -11  -26   23  -13]
  [  -4  -17  -74  -68   15   98   -3   72  -41  -43   78   73   32   85
     -7   -9   45   63   94   48]
  [  73   85   19   64    5   90  -96   58    9  -13   63  -27   83  -74
     18  -78   -2  -10  -49  -54]
  [ -39   88  -53    4   28   38   41  -29   -6  -94   73   58  -85   69
     66  -47   71  -18   35  -35]
  [  69  -34   14   -8  -22    0   59   78   74   -7   14   61  -88  -20
    -34   25   38   12   55   84]
  [  20  -35   92   97  -12  -66  -97   88   65   71  -12  -30   48   34
    -72   15   34  -34   -8    2]
  [   1   23   97    9  -27    0   82  -23   49   59  -19  -65   36  -75
    -79   73   44   53   19   65]
  [  27   29   33   98   40  -10  -26   82  -22  -38  -28   99  -55   33
    -53   87   70   95   38  -43]
  [ -11   31   25  -18   97   86   32  -83   97   91   -6   52   31  -31
     68   64  -42   77   83   52]
  [  61   46   -3   35   81  -54   27   61  -19   57  -88   18  -54   18
    -68  -66   15  -13   24   53]]

 [[  74    7  -48   10  -36  -24   18  -27   46  -96  -93   31  -52  -24
    -63   16  -52    9   79  -51]
  [ -29  -10   77  -71  -97  -26  -16  -35   19  -80  -58  -72   54  -13
     80  -52   94    2  -91  -98]
  [  16    8  -11   24   17    0  -10  -32    5  -90  -88    4   65   67
     60   22  -67   54   -1  -50]
  [ -12  -80   56  -28  -82   53   52  -61   23  -72   52   46   95   72
    -55  -46 -100   38   95   34]
  [  -1   85   98   47   62  -50    2  -80   71   18   28    9    5   94
     40  -85   18  -67   51   40]
  [  33  -62  -99  -90  -94   25  -94    2   66  -25   14  -15   26   14
     78  -49  -98  -24   57  -91]
  [  15   33   16   81   93  -57  -21  -38  -24  -53  -24   49   20  -82
    -11  -87   -8    5  -94    0]
  [ -52  -60   54   58   33   43   91  -63  -41   13   -5   -4  -68  -38
     38   40   -7  -36   57  -10]
  [  50   47   33  -64  -17   27  -95   -8   54  -17   40   35   80  -20
     57 -100  -26   31  -39  -33]
  [  44   12  -16   16  -44  -38   84  -52   45   17   85  -91   90  -21
     61   32   44  -29  -38  -77]]

 [[  -9   10  -93  -78   -2   58  -20   55  -53   14  -89  -22  -70  -10
    -66  -46  -19   63  -29    9]
  [ -41   12   65  -32   81  -87    1  -81    7  -92   -4   97   24    0
     29   57   33   68   50  -70]
  [ -81  -96   48    6  -26   23   95  -40   -7   69  -60   88   79  -60
    -41  -71   -6   65   26  -84]
  [  -1   67   57  -35  -77   28  -13  -63   11   91   54  -11   34    1
    -59   45   12  -57   10   97]
  [  18   47  -78    9   39  -89   61   35   19  -74  -52   99   82   -4
      0  -18  -13   49  -98  -92]
  [ -90  -95  -62   66    0   93   17  -41   64   33  -95  -62   63  -12
     77  -16   14  -91   32   77]
  [ -76   -6   30  -17   31  -23  -89   41  -19   54   98   75   -2  -79
     48   70   22   85   45    1]
  [  83    0   96   11  -89   -3   47   12  -89  -75   -3   -5  -55  -94
    -11  -12  -62  -49  -84   51]
  [ -97  -10   74   22   57  -98   33   21   99  -85  -22   63   80    3
     18  -93   79    2   79   57]
  [  83   13   39   95   22  -45  -12  -32   17   15   85   -7    2   39
    -18  -97   65   35  -71  -22]]

 [[ -89  -89  -84  -40   23    3   91   87   29   46   81  -72   92  -15
    -27   36   39   17   79  -19]
  [  83  -85   31    6  -72  -42  -22   11  -35  -24  -89  -75    3  -89
    -10   62   29   44  -99  -84]
  [ -67  -67   72  -60  -28    6  -17   60   51  -32   59   50  -36  -69
    -21  -17  -85  -49   40   73]
  [ -90    5  -20  -30  -79   95  -20  -36   29  -50   -4    7  -18   85
     50  -85   43  -72  -29  -73]
  [ -43  -42  -87   46  -22  -80  -29   83  -56   -9  -56  -85  -13  -23
     57   -5   10   32  -72   93]
  [ -51   77  -13  -43  -59   94   75  -83  -80   66  -36   34   50  -21
    -26   62   68   66   49  -66]
  [  17   60   70   27  -56   -1  -59    3   55  -52   27   38  -32  -83
    -97    1   -6  -71    2   23]
  [  58   94  -40   35   79  -27   92   45   68  -79   -6   54   43  -83
    -90   45   31  -27  -71   95]
  [  99   32   89  -10    0   34  -68  -19   19   18  -63   19  -73  -49
    -22   87  -14   -5  -92  -44]
  [ -71   56   62   86   27   26   11   44  -41  -93   40  -68  -25  -60
   -100    9   -8   65   75  -39]]]
h: [[  3  78 -32  85  19  32   5 -64 -20  65]
 [ 17 -65  76  28 -51  85 -91 -50  76 -88]
 [ 98  24  64  -1   2 -64 -70  14  47  66]
 [ 72 -65 -86 -71  79 -40 -19 -33 -71  55]]""")
#  你调用工具时候传入的参数格式是一个整数列表维度跟x的维度匹配，例如：[1,0,1,0,1,0,1,0,1,0]代表x的取值。

    test_message = HumanMessage("Hello!")
    
    # res = agent.invoke(
    #     {"messages": [input_message]}, # may have other states stored in memory 
    #     config=config, 
    # )

    # print(type(res))
    # print(res)


    

    for chunk in agent.stream(
        {"messages": [input_message]},
        stream_mode="updates",
        config=config, 
    ): 
        for _, data in chunk.items():
            # print(f"step: {step}")
            # print(f"content: {data['messages'][-1].content_blocks}") 
            message = data['messages'][-1]
            reasoning_content = message.additional_kwargs.get("reasoning_content", "") 
            print("---" * 33)
            print(f"Reasoning content: {reasoning_content}")
            message.pretty_print()
            print("---" * 33)

    # for stream_mode, data in agent.stream(
    #     input_message,
    #     stream_mode=["messages", "updates"],
    #     config=config, 
    # ):
    #     if stream_mode == "messages":
    #         print(stream_mode, data)
    #         # token, metadata = data
    #         # if isinstance(token, AIMessageChunk):
    #         #     _render_message_chunk(token)  
    #     if stream_mode == "updates":
    #         print(stream_mode, data)
    #         # for source, update in data.items():
    #         #     if source in ("model", "tools"):  # `source` captures node name
    #         #         _render_completed_message(update["messages"][-1])  
