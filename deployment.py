import json
import time
import uuid
from typing import Any, Optional, Tuple

from langchain.prompts import ChatPromptTemplate
from langchain.schema.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import create_react_agent
from pydantic import SecretStr
from data.State import DeploymentState, ElementMatch
from data.graph_db import Neo4jDatabase
from data.vector_db import VectorStore
from tool.img_tool import *
from tool.screen_content import *

os.environ["LANGCHAIN_TRACING_V2"] = config.LANGCHAIN_TRACING_V2
os.environ["LANGCHAIN_ENDPOINT"] = config.LANGCHAIN_ENDPOINT
os.environ["LANGCHAIN_API_KEY"] = config.LANGCHAIN_API_KEY
os.environ["LANGCHAIN_PROJECT"] = "DeploymentExecution"

model = ChatOpenAI(
    openai_api_base=config.LLM_BASE_URL,
    openai_api_key=SecretStr(config.LLM_API_KEY),
    model_name=config.LLM_MODEL,
    request_timeout=config.LLM_REQUEST_TIMEOUT,
    max_retries=config.LLM_MAX_RETRIES,
    max_tokens=config.LLM_MAX_TOKEN,
)

URI = config.Neo4j_URI
AUTH = config.Neo4j_AUTH
db = Neo4jDatabase(URI, AUTH)

vector_db = VectorStore(api_key=config.PINECONE_API_KEY)


def create_execution_state(device: str) -> Dict[str, Any]:
    """
    Create initial execution state

    Args:
        device: Device ID

    Returns:
        Dictionary containing initial state
    """
    from data.State import create_deployment_state

    state = create_deployment_state(
        task="",
        device=device,
    )

    return state


def match_task_to_action(
    state: Dict[str, Any], task: str
) -> Tuple[bool, Optional[Dict[str, Any]]]:
    """
    Match user task with high-level action nodes

    Args:
        state: Execution state
        task: User input task description

    Returns:
        (whether match successful, matched high-level action node)
    """
    print(f"Matching task: {task}")

    # 1. Get all high-level action nodes from database
    high_level_actions = db.get_all_high_level_actions()

    if not high_level_actions:
        print("❌ No high-level action nodes found")
        return False, None

    print(f"Found {len(high_level_actions)} high-level action nodes")

    if len(high_level_actions) == 0:
        return False, None

    # 2. Create task matching prompt
    task_match_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are an AI assistant specialized in matching user tasks with predefined high-level actions.
You need to analyze the user's task description and determine if it matches any predefined high-level actions.
If you find a matching high-level action, return the complete information of that action. If no match is found, clearly indicate no match.
Only consider it a match when the matching degree is high (above 0.7).""",
            ),
            (
                "human",
                """User task: {task}

Available high-level actions:
{actions_json}

Please determine if the user task matches any high-level action.
If matched successfully, return the complete information of the best matching action (keeping the original JSON format) with "MATCHED: " prefix.
If no match is found, return "NO_MATCH" with a brief explanation.
""",
            ),
        ]
    )

    # 3. Prepare JSON string of high-level actions
    actions_json = json.dumps(high_level_actions, ensure_ascii=False, indent=2)

    # 4. Call LLM for matching
    try:
        # Prepare input
        match_input = {"task": task, "actions_json": actions_json}

        # Use simple string output parser
        match_chain = task_match_prompt | model | StrOutputParser()

        # Execute matching
        result = match_chain.invoke(match_input)

        # Parse results
        if result.startswith("MATCHED:"):
            # Extract matched action information
            action_json_str = result[len("MATCHED:") :].strip()
            try:
                matched_action = json.loads(action_json_str)
                print(
                    f"✓ Found matching high-level action: {matched_action.get('name', 'Unknown')} (ID: {matched_action.get('action_id', 'Unknown')})"
                )
                return True, matched_action
            except json.JSONDecodeError:
                print(f"❌ Cannot parse matching result: {action_json_str}")
                return False, None
        elif result.startswith("NO_MATCH"):
            reason = result[len("NO_MATCH") :].strip()
            print(f"❌ No matching high-level action found")
            print(f"  Reason: {reason}")
            return False, None
        else:
            print(f"❌ Unrecognized matching result: {result}")
            return False, None

    except Exception as e:
        print(f"❌ Error during task matching: {str(e)}")
        return False, None


def capture_and_parse_screen(state: DeploymentState) -> DeploymentState:
    """
    Capture current screen and parse elements, update state

    Args:
        state: Deployment state

    Returns:
        Updated deployment state
    """
    try:
        # 1. Take screenshot
        screenshot_path = take_screenshot.invoke(
            {
                "device": state["device"],
                "app_name": "deployment",
                "step": state["current_step"],
            }
        )

        if not screenshot_path or not os.path.exists(screenshot_path):
            print("❌ Screenshot failed")
            return state

        # 2. Parse screen elements
        screen_result = screen_element.invoke({"image_path": screenshot_path})

        if "error" in screen_result:
            print(f"❌ Screen element parsing failed: {screen_result['error']}")
            return state

        # 3. Update current page information
        state["current_page"]["screenshot"] = screenshot_path
        state["current_page"]["elements_json"] = screen_result[
            "parsed_content_json_path"
        ]

        # 4. Load element data
        with open(
            screen_result["parsed_content_json_path"], "r", encoding="utf-8"
        ) as f:
            state["current_page"]["elements_data"] = json.load(f)

        print(
            f"✓ Successfully parsed current screen, detected {len(state['current_page']['elements_data'])} UI elements"
        )
        return state

    except Exception as e:
        print(f"❌ Error capturing and parsing screen: {str(e)}")
        return state


def match_screen_elements(
    state: DeploymentState, action_sequence: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Match current screen elements with elements in high-level action nodes using visual embedding comparison

    Args:
        state: Deployment state
        action_sequence: Element sequence in high-level action nodes

    Returns:
        List of matching results, including screen element ID and matching score
    """
    if not state["current_page"]["elements_data"]:
        print("❌ Current screen element data is empty")
        return []

    # Get current step information
    current_step_idx = state["current_step"]
    if current_step_idx >= len(action_sequence):
        print(f"⚠️ Current step {current_step_idx} exceeds action sequence range")
        return []

    current_action = action_sequence[current_step_idx]

    # Get element information
    element_id = current_action.get("element_id")
    if not element_id:
        print("⚠️ No element ID specified in current step")
        return []

    # 首先尝试从action nodes获取元素
    db_element = db.get_action_by_id(element_id)
    if not db_element:
        print(f"⚠️ [1/2] 在action nodes中未找到元素 ID: {element_id}, 尝试从普通element nodes获取...")
        # 如果action nodes中未找到，则尝试从普通element nodes获取
        db_element = db.get_element_by_id(element_id)
        if not db_element:
            print(f"❌ [2/2] 在普通element nodes中仍未找到元素 ID: {element_id}")
            print(f"     元素ID: {element_id}")
            print(f"     当前步骤: {current_action}")
            return []
        else:
            print(f"✅ [2/2] 成功从普通element nodes获取到元素 ID: {element_id}")
    else:
        print(f"✅ [1/1] 成功从action nodes获取到元素 ID: {element_id}")
        
    # 打印元素信息用于调试
    if db_element:
        print(f"📋 元素信息:")
        print(f"   - 类型: {db_element.get('type', 'N/A')}")
        print(f"   - 内容: {db_element.get('content', 'N/A')}")
        print(f"   - 是否有视觉特征: {'是' if db_element.get('visual_embedding') else '否'}")
        print(f"   - 截图路径: {db_element.get('screenshot_path', 'N/A')}")
        
        # 打印完整的元素对象信息
        print("\n🔍 完整元素对象:")
        for key, value in db_element.items():
            # 如果值是字节类型，则只显示类型和长度
            if isinstance(value, (bytes, bytearray)):
                print(f"   - {key}: <{type(value).__name__} 长度={len(value)}>")
            # 如果值太长则截断显示
            elif isinstance(value, str) and len(value) > 100:
                print(f"   - {key}: {value[:100]}... (共{len(value)}字符)")
            # 其他情况正常显示
            else:
                print(f"   - {key}: {value}")

    # If retrieved node is an Action node, ensure it contains necessary visual information
    # Otherwise fall back to semantic matching
    if "action_id" in db_element and not any(
        key in db_element for key in ["visual_embedding", "screenshot_path"]
    ):
        print(
            f"⚠️ Retrieved node is an Action node but lacks visual information, falling back to semantic matching"
        )
        return fallback_to_semantic_match(state, action_sequence)

    # Check if visual embedding exists
    template_embedding = None
    if "visual_embedding" in db_element and db_element["visual_embedding"]:
        template_embedding = db_element["visual_embedding"]
    else:
        print("⚠️ Template element has no visual embedding, trying to extract features")
        # Try to get element screenshot or extract features using bounding box information
        if "screenshot_path" in db_element and db_element["screenshot_path"]:
            try:
                # Extract template element features from local screenshot
                print("🔄 从本地截图提取视觉特征...")
                template_embedding = extract_features(
                    db_element["screenshot_path"], "resnet50"
                )["features"]
                print("✅ 成功从本地截图提取视觉特征")
            except Exception as e:
                print(f"❌ 无法从本地截图提取特征: {str(e)}")
                print("🔄 尝试从Pinecone获取特征...")
                template_embedding = None
        else:
            print("ℹ️ 未找到本地截图路径")
            template_embedding = None
            
        # 如果本地特征提取失败或没有本地截图，尝试从Pinecone获取
        if template_embedding is None:
            try:
                print(f"🔄 尝试从Pinecone获取元素 {element_id} 的视觉特征...")
                # 从Pinecone获取元素特征
                result = vector_db.index.fetch(
                    ids=[element_id],
                    namespace="element"  # 确保与存储时的命名空间一致
                )
                
                if element_id in result.vectors:
                    vector_data = result.vectors[element_id]
                    template_embedding = vector_data.values
                    print(f"✅ 成功从Pinecone获取视觉特征 (维度: {len(template_embedding)})")
                    
                    # 调试信息
                    print(f"📌 元素元数据: {vector_data.metadata}")
                else:
                    print(f"❌ 在Pinecone中未找到元素 {element_id} 的特征")
                    return fallback_to_semantic_match(state, action_sequence)
            except Exception as e:
                print(f"❌ 从Pinecone获取特征时出错: {str(e)}")
                print("🔄 回退到语义匹配...")
                return fallback_to_semantic_match(state, action_sequence)

    # Process current screen elements
    screen_elements = state["current_page"]["elements_data"]
    screenshot_path = state["current_page"]["screenshot"]
    elements_json_path = state["current_page"]["elements_json"]

    try:
        # Get visual embeddings for all elements on current screen
        from tool.img_tool import elements_img, extract_features

        print(f"\n🔍 Starting feature extraction for {len(screen_elements)} screen elements...")
        element_embeddings = []
        
        for idx, element in enumerate(screen_elements):
            element_id = element.get("ID", idx)
            # Print detailed element information
            element_name = element.get('name', 'No name')
            element_text = element.get('text', 'No text')
            element_class = element.get('class', 'No class')
            element_bounds = element.get('bounds', 'No bounds')
            element_visible = element.get('visible', 'Unknown')
            
            print(f"\n🔄 Processing element {idx} (ID: {element_id})")
            print(f"  📝 Element details:")
            print(f"     - Name: {element_name}")
            print(f"     - Text: {element_text}")
            print(f"     - Class: {element_class}")
            print(f"     - Bounds: {element_bounds}")
            print(f"     - Visible: {element_visible}")
            print(f"     - All properties: {element}")
            
            try:
                # Prepare input for elements_img with correct parameter names
                input_data = {
                    "page_path": screenshot_path,  # Changed from screenshot_path to page_path
                    "json_path": elements_json_path,  # Using the elements_json_path from state
                    "IDs": [str(element_id)]  # Changed from element_id to IDs and made it a list
                }
                
                print(f"  📂 Input data prepared for elements_img")
                
                # Get element image using invoke
                print("  🖼️  Calling elements_img.invoke()...")
                element_img_stream = elements_img.invoke(input_data)
                
                if not element_img_stream:
                    print(f"  ⚠️  Empty response from elements_img for element {idx}")
                    continue
                    
                print(f"  ✅ Successfully got element image (type: {type(element_img_stream)})")
                
                # Extract features
                print("  🔄 Extracting visual features...")
                element_feature = extract_features(element_img_stream, "resnet50")
                
                if not element_feature or "features" not in element_feature:
                    print(f"  ⚠️  Failed to extract features for element {idx}")
                    continue
                    
                feature_vector = element_feature["features"]
                print(f"  ✅ Extracted features (dimensions: {len(feature_vector) if feature_vector else 0})")
                
                element_embeddings.append((idx, feature_vector))
                print(f"  🎯 Added features for element {idx} to processing queue")
                
            except Exception as e:
                print(f"  ❌ Error processing element {idx}: {str(e)}")
                import traceback
                traceback.print_exc()
                continue

        if not element_embeddings:
            print("\n❌ Failed to extract features for any screen elements")
            return fallback_to_semantic_match(state, action_sequence)
            
        print(f"\n✅ Successfully extracted features for {len(element_embeddings)}/{len(screen_elements)} elements")

        # Calculate similarity and sort
        import numpy as np

        matches = []
        for idx, embedding in element_embeddings:
            # Calculate cosine similarity
            template_vec = np.array(template_embedding).flatten()
            element_vec = np.array(embedding).flatten()

            # Normalize vectors
            template_norm = np.linalg.norm(template_vec)
            element_norm = np.linalg.norm(element_vec)

            if template_norm == 0 or element_norm == 0:
                similarity = 0
            else:
                similarity = np.dot(template_vec, element_vec) / (
                    template_norm * element_norm
                )

            # Convert similarity to match score
            match_score = float(similarity)

            if match_score >= 0.6:  # Matching threshold
                matches.append(
                    {
                        "element_id": element_id,
                        "match_score": match_score,
                        "screen_element_id": idx,
                        "action_type": current_action.get("atomic_action", "tap"),
                        "parameters": current_action.get("action_params", {}),
                    }
                )

        # Sort by similarity
        matches.sort(key=lambda x: x["match_score"], reverse=True)

        if matches:
            best_match = matches[0]
            matched_element = next((e for e in screen_elements if str(e.get('ID')) == str(best_match['screen_element_id'])), None)
            print(f"\n🎯 MATCH FOUND!")
            print(f"✓ Element ID: {best_match['screen_element_id']}")
            print(f"✓ Match score: {best_match['match_score']}")
            print(f"✓ Action type: {best_match['action_type']}")
            print("\n🔍 Matched element details:")
            if matched_element:
                for k, v in matched_element.items():
                    if k not in ['screenshot', 'screenshot_path']:  # Skip binary data
                        print(f"   - {k}: {v}")
            else:
                print("   Element details not found")
                
            # Add debug info about the match
            print("\n🔍 Match details:")
            print(f"   - Screen element ID: {best_match['screen_element_id']}")
            print(f"   - Template element ID: {best_match.get('template_element', {}).get('id', 'N/A')}")
            print(f"   - Action type: {best_match.get('action_type', 'tap')}")
            print(f"   - Match score: {best_match.get('match_score', 0):.4f}")
            
            # Store the best match in state for later use
            state['last_matched_element'] = best_match
            
            return matches
        else:
            print("❌ No matching screen element found")
            return []

    except Exception as e:
        print(f"❌ Error during visual matching: {str(e)}")
        # Fall back to semantic matching on error
        return fallback_to_semantic_match(state, action_sequence)
    

def fallback_to_semantic_match(
    state: DeploymentState, action_sequence: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Fallback to semantic matching when visual matching fails
    """
    print("🔄 Falling back to semantic matching...")

    # Prepare template element features
    template_elements = []
    current_step_idx = state["current_step"]
    if current_step_idx >= len(action_sequence):
        return []

    step_info = action_sequence[current_step_idx]

    # Get element information
    element_id = step_info.get("element_id")
    if not element_id:
        return []

    # Get element information from database - using correct method name
    db_element = db.get_action_by_id(element_id)
    if not db_element:
        print(f"⚠️ Element with ID {element_id} not found")
        # Try to get from another type
        db_element = db.get_element_by_id(element_id)
        if not db_element:
            print(f"⚠️ Action with ID {element_id} also not found")
            return []

    template_elements.append({"db_element": db_element, "step_info": step_info})

    # If no elements to match, return empty list
    if not template_elements:
        return []

    current_template = template_elements[0]

    # Prepare matching prompt
    element_match_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are an AI assistant specialized in matching UI elements. You need to analyze template element descriptions and current screen elements to find the best match. Your answer must be in JSON format, including the matching results.""",
            ),
            (
                "human",
                """Template element description: 
{template_element}

Current screen elements:
{screen_elements}

Please find the screen element that best matches the template element, and return in the following JSON format:
{{
  "element_id": "{element_id}",
  "match_score": matching score (0-1),
  "screen_element_id": screen element ID,
  "action_type": "atomic action type (tap/text/swipe etc.)",
  "parameters": {{action parameters object}}
}}

If no element with matching score above 0.6 is found, set match_score to 0 and screen_element_id to -1.
""",
            ),
        ]
    )
    print("[调试] element_match_prompt.input_variables:", element_match_prompt.input_variables)

    # Parse template element information
    current_db_element = current_template["db_element"]

    # Determine correct ID field
    element_id_field = (
        "element_id" if "element_id" in current_db_element else "action_id"
    )
    template_element_desc = (
        f"ID: {current_db_element.get(element_id_field, 'unknown')}\n"
    )

    if "description" in current_db_element and current_db_element["description"]:
        template_element_desc += f"Description: {current_db_element['description']}\n"
    elif "name" in current_db_element and current_db_element["name"]:
        template_element_desc += f"Name: {current_db_element['name']}\n"

    # Check position information, supporting different field names
    bbox_field = None
    for field in ["bounding_box", "bbox", "position"]:
        if field in current_db_element and current_db_element[field]:
            bbox_field = field
            break

    if bbox_field:
        bbox = current_db_element[bbox_field]
        if isinstance(bbox, list) and len(bbox) >= 4:
            template_element_desc += f"Position: [{bbox[0]:.3f}, {bbox[1]:.3f}, {bbox[2]:.3f}, {bbox[3]:.3f}]\n"
        elif isinstance(bbox, str):
            template_element_desc += f"Position: {bbox}\n"

    # Add action information
    template_element_desc += (
        f"Action type: {current_template['step_info'].get('atomic_action', 'tap')}\n"
    )

    if (
        "action_params" in current_template["step_info"]
        and current_template["step_info"]["action_params"]
    ):
        params = current_template["step_info"]["action_params"]
        if isinstance(params, str):
            try:
                params = json.loads(params)
            except:
                pass

        if isinstance(params, dict):
            template_element_desc += "Action parameters:\n"
            for k, v in params.items():
                template_element_desc += f"  {k}: {v}\n"

    # Parse screen elements information
    screen_elements_desc = ""
    for i, element in enumerate(state["current_page"]["elements_data"]):
        screen_elements_desc += f"Element {i} (ID: {element.get('ID', i)}):\n"
        if "type" in element:
            screen_elements_desc += f"  Type: {element['type']}\n"
        if "content" in element:
            screen_elements_desc += f"  Content: {element['content']}\n"
        if "bbox" in element:
            bbox = element["bbox"]
            screen_elements_desc += f"  Position: [{bbox[0]:.3f}, {bbox[1]:.3f}, {bbox[2]:.3f}, {bbox[3]:.3f}]\n"
        screen_elements_desc += "\n"

    # Call LLM for matching
    try:
        # Prepare input
        match_input = {
            "element_id": current_db_element.get(element_id_field, "unknown"),
            "template_element": template_element_desc,
            "screen_elements": screen_elements_desc,
        }

        # Create parser
        parser = JsonOutputParser(pydantic_object=ElementMatch)

        # Build chain
        match_chain = element_match_prompt | model | parser

        # Execute matching with detailed logging
        print("\n=== Starting LLM Matching ===")
        print("Match Input:", json.dumps(match_input, indent=2, ensure_ascii=False))
        
        try:
            match_result = match_chain.invoke(match_input)
            print("\nMatch Result (Raw):", match_result)
            
            # Convert to dict for better logging
            result_dict = match_result if isinstance(match_result, dict) else match_result.dict()
            print("\nMatch Result (Parsed):", json.dumps(result_dict, indent=2, ensure_ascii=False))
            
            # 安全地获取匹配分数和元素ID
            match_score = match_result.get('match_score', 0) if isinstance(match_result, dict) else getattr(match_result, 'match_score', 0)
            screen_element_id = match_result.get('screen_element_id', -1) if isinstance(match_result, dict) else getattr(match_result, 'screen_element_id', -1)
            
            if match_score >= 0.6 and screen_element_id >= 0:
                print("\n✓ Match Successful!")
                print(f"  Screen Element ID: {screen_element_id}")
                print(f"  Match Score: {match_score}")
                
                # 安全获取 action_type
                action_type = match_result.get('action_type', 'tap') if isinstance(match_result, dict) else getattr(match_result, 'action_type', 'tap')
                print(f"  Action Type: {action_type}")
                
                # 安全获取参数
                params = match_result.get('parameters', {}) if isinstance(match_result, dict) else getattr(match_result, 'parameters', {})
                
                if params:
                    print(f"  Parameters: {params}")
                
                # 准备返回结果
                result = {
                    'element_id': match_result.get('element_id', '') if isinstance(match_result, dict) else getattr(match_result, 'element_id', ''),
                    'match_score': match_score,
                    'screen_element_id': screen_element_id,
                    'action_type': action_type,
                    'parameters': params if isinstance(params, dict) else {}
                }
                return [result]
            else:
                print("\n❌ No matching screen element found (score too low or invalid ID)")
                return []
                
        except Exception as e:
            print(f"\n❌ Error during LLM matching: {str(e)}")
            import traceback
            traceback.print_exc()
            return []
            
    except Exception as e:
        print(f"\n❌ Error preparing LLM matching: {str(e)}")
        import traceback
        traceback.print_exc()
        return []


def debug_print(*args, **kwargs):
    """Helper function for debug output"""
    print("\n" + "="*80)
    print("🐛 [DEBUG]", *args, **kwargs)
    print("="*80 + "\n")


def execute_element_action(state: DeploymentState, element_match: Dict[str, Any]) -> bool:
    """
    Execute screen element action

    Args:
        state: Execution state
        element_match: Element matching result

    Returns:
        Whether the operation was successful
    """
    # 在函数开始处导入所有需要的模块
    import time as time_module
    import json
    
    try:
        current_time = time_module.strftime('%Y-%m-%d %H:%M:%S')
        
        # 记录函数开始执行
        execution_id = str(uuid.uuid4())[:8]  # 生成一个简短的执行ID用于日志跟踪
        print("\n" + "="*80)
        print(f"🚀 [执行ID:{execution_id}] EXECUTE_ELEMENT_ACTION 函数被调用")
        print(f"📌 开始时间: {current_time}")
        print("\n🔍 元素匹配结果:")
    except Exception as e:
        print(f"❌ 初始化错误: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    print(json.dumps(element_match, indent=2, ensure_ascii=False, default=str))
    
    # 记录state基本信息
    print("\n📊 State 基本信息:")
    print(f"  - 包含的键: {list(state.keys())}")
    print(f"  - 设备: {state.get('device', '未指定')}")
    print(f"  - 当前步骤: {state.get('current_step', 0)}/{state.get('total_steps', 1)}")
    
    # 检查关键数据结构
    if 'current_page' not in state:
        print("❌ 错误: state 中缺少 'current_page' 键")
        return False
    
    current_page = state['current_page']
    print("\n📄 当前页面信息:")
    print(f"  - 页面元素数量: {len(current_page.get('elements_data', []))}")
    print(f"  - 截图路径: {current_page.get('screenshot', '未指定')}")
    
    # 检查screen_action工具
    print("\n🔧 工具检查:")
    print(f"  - screen_action 类型: {type(screen_action).__name__}")
    print(f"  - screen_action 可调用: {callable(screen_action.invoke)}")
    print(f"  - screen_action 模块: {screen_action.__module__}")
    
    # 记录内存使用情况
    try:
        import psutil
        process = psutil.Process()
        mem_info = process.memory_info()
        print(f"\n💾 内存使用: {mem_info.rss / 1024 / 1024:.2f} MB")
    except ImportError:
        print("\nℹ️ 安装psutil包可以查看内存使用情况")
    
    print("\n" + "="*80)
    print("🛠️ 开始执行元素操作...")
    
    try:
        # 记录函数开始时间
        func_start_time = time_module.time()
        
        if not element_match:
            print(f"❌ [{execution_id}] 错误: 元素匹配结果为空")
            return False
            
        # 记录操作详情
        action_type = element_match.get("action_type", "tap").upper()
        print(f"\n🎯 准备执行 {action_type} 操作:")
        print(f"  - 元素ID: {element_match.get('element_id', 'N/A')}")
        print(f"  - 元素类型: {element_match.get('element_type', 'N/A')}")
        print(f"  - 元素内容: {element_match.get('element_content', 'N/A')}")
        print(f"  - 匹配分数: {element_match.get('match_score', 0):.2f}")
        
        # 记录边界框信息
        bbox = element_match.get('bbox', [])
        if bbox and len(bbox) == 4:
            print(f"  - 元素位置: x1={bbox[0]:.2f}, y1={bbox[1]:.2f}, x2={bbox[2]:.2f}, y2={bbox[3]:.2f}")
        
        print("\n🔄 准备执行操作...")
        element_id = element_match.get("element_id")
        
        print("\n📌 操作详情:")
        print(f"  操作类型: {action_type}")
        print(f"  元素ID: {element_id}")
        print(f"  屏幕元素ID: {element_match.get('screen_element_id', 'N/A')}")
        print(f"  参数: {json.dumps(element_match.get('parameters', {}), ensure_ascii=False, indent=2, default=str)}")
        
        # 验证必要参数
        if not element_id and element_id != 0:  # 允许element_id为0
            print("❌ 错误: 缺少元素ID")
            return False
        
        # 记录当前页面元素信息
        current_page = state['current_page']
        elements_data = current_page.get('elements_data', [])
        print(f"\n📋 当前页面元素信息:")
        print(f"  - 页面元素总数: {len(elements_data)}")
        
        # 查找当前元素在页面元素列表中的位置
        screen_element = None
        screen_element_id = element_match.get("screen_element_id", -1)
        
        # 记录元素查找过程
        print(f"\n🔍 查找元素 (ID: {element_id}, 屏幕ID: {screen_element_id}):")
        
        # 优先使用screen_element_id查找
        if screen_element_id != -1 and 0 <= screen_element_id < len(elements_data):
            screen_element = elements_data[screen_element_id]
            print(f"  ✅ 通过screen_element_id找到元素")
        # 否则尝试通过element_id查找
        elif element_id is not None:
            for idx, elem in enumerate(elements_data):
                if elem.get("ID") == element_id or idx == element_id:
                    screen_element = elem
                    screen_element_id = idx
                    print(f"  ✅ 通过element_id找到元素 (索引: {idx})")
                    break
        
        if not screen_element:
            print(f"❌ 错误: 未找到匹配的元素 (ID: {element_id}, 屏幕ID: {screen_element_id})")
            print(f"     页面元素ID列表: {[e.get('ID', 'N/A') for e in elements_data[:10]]}{'...' if len(elements_data) > 10 else ''}")
            return False
            
        # 记录找到的元素信息
        print(f"  - 元素类型: {screen_element.get('type', 'unknown')}")
        print(f"  - 元素内容: {screen_element.get('content', 'N/A')}")
        print(f"  - 可交互: {screen_element.get('interactivity', False)}")
        
        # 记录边界框信息
        bbox = screen_element.get('bbox', [])
        if bbox and len(bbox) == 4:
            print(f"  - 元素位置: x1={bbox[0]:.2f}, y1={bbox[1]:.2f}, x2={bbox[2]:.2f}, y2={bbox[3]:.2f}")
            
        print("\n🔄 准备执行操作...")
        element_id = element_match.get("element_id", "unknown")
        
        print("\n📌 操作详情:")
        print(f"  操作类型: {action_type}")
        print(f"  元素ID: {element_id}")
        print(f"  屏幕元素ID: {screen_element_id}")
        print(f"  参数: {json.dumps(element_match.get('parameters', {}), ensure_ascii=False, indent=2, default=str)}")
        
        # 检查当前页面数据
        if "current_page" not in state:
            print("❌ 错误: state 中缺少 'current_page' 键")
            return False
            
        if "elements_data" not in state["current_page"]:
            print("❌ 错误: state['current_page'] 中缺少 'elements_data' 键")
            return False

        # 验证屏幕元素ID是否有效
        elements_data = state["current_page"]["elements_data"]
        elements_count = len(elements_data)
        print(f"\n🔍 页面元素验证:")
        print(f"  当前页面元素数量: {elements_count}")
        print(f"  请求的屏幕元素ID: {screen_element_id}")
        
        if not isinstance(screen_element_id, int) or screen_element_id < 0 or screen_element_id >= elements_count:
            print(f"❌ 错误: 无效的屏幕元素ID: {screen_element_id} (有效范围: 0-{elements_count-1})")
            if elements_count > 0:
                print("  可用的元素ID示例:")
                for i in range(min(3, elements_count)):  # 只显示前3个元素作为示例
                    print(f"    元素 {i}: {elements_data[i].get('text', elements_data[i].get('content', 'No content'))}")
            return False

        # 获取元素位置
        print("\n🔍 获取元素位置信息...")
        element = elements_data[screen_element_id]
        bbox = element.get("bbox", [0, 0, 0, 0])
        print(f"  元素边界框 (归一化坐标): {bbox}")
        print(f"  元素内容: {element.get('content', element.get('text', '无内容'))}")
        print(f"  元素类型: {element.get('type', '未知')}")
        print(f"  元素属性: {json.dumps({k: v for k, v in element.items() if k not in ['bbox', 'content', 'text', 'type']}, default=str)}")

        # 获取设备尺寸并计算中心点
        print("\n📱 获取设备尺寸...")
        try:
            device_size = get_device_size.invoke(state["device"])
            if isinstance(device_size, str):
                print(f"⚠️ 使用默认设备尺寸 (获取失败: {device_size})")
                device_size = {"width": 1080, "height": 1920}
            else:
                print(f"  设备尺寸: {device_size.get('width', 'N/A')}x{device_size.get('height', 'N/A')}")
        except Exception as e:
            print(f"❌ 获取设备尺寸时出错: {str(e)}")
            device_size = {"width": 1080, "height": 1920}
            print(f"⚠️ 使用默认设备尺寸: {device_size['width']}x{device_size['height']}")
            
        # 计算点击位置（归一化坐标转换为实际像素）
        try:
            center_x = int((float(bbox[0]) + float(bbox[2])) / 2 * float(device_size["width"]))
            center_y = int((float(bbox[1]) + float(bbox[3])) / 2 * float(device_size["height"]))
            print(f"🎯 计算点击位置: ({center_x}, {center_y})")
            print(f"  基于边界框: {bbox}")
            print(f"  使用设备尺寸: {device_size['width']}x{device_size['height']}")
        except (IndexError, ValueError, TypeError) as e:
            print(f"❌ 计算点击位置时出错: {str(e)}")
            print(f"  边界框: {bbox}")
            print(f"  设备尺寸: {device_size}")
            return False

        # 准备操作参数
        debug_print("准备操作参数...")
        action_params = {
            "action_type": action_type,
            "element_id": screen_element_id,
            "x": center_x,  # 添加 x 坐标
            "y": center_y,  # 添加 y 坐标
            **element_match.get('parameters', {})  # 保留原始参数
        }
        
        debug_print(f"操作参数: {json.dumps(action_params, indent=2, ensure_ascii=False, default=str)}")
        # 安全地获取device和action参数
        device_info = state.get('device', '未指定设备')
        action_info = action_params.get('action_type', '未知操作')
        print(f"  基础参数: device={device_info}, action={action_info}")
        print(f"  坐标: x={center_x}, y={center_y}")

        # 确保action_params包含必要的键
        action_params['device'] = state.get('device')
        action_params['action'] = action_type.lower()

        # 根据操作类型添加特定参数
        print("\n🔧 设置操作特定参数...")
        if action_type == "text":
            text = element_match.get('parameters', {}).get("text", "")
            action_params["input_str"] = text
            print(f"  ⌨️ 文本输入: '{text}'")
        elif action_type == "long_press":
            duration = element_match.get('parameters', {}).get("duration", 1000)
            action_params["duration"] = duration
            print(f"  ⏱️ 长按时长: {duration}ms")
        elif action_type == "swipe":
            direction = element_match.get('parameters', {}).get("direction", "up")
            distance = element_match.get('parameters', {}).get("distance", "medium")
            action_params["direction"] = direction
            action_params["dist"] = distance
            print(f"  🔄 滑动方向: {direction}, 距离: {distance}")
        else:
            print(f"  ℹ️ 基本点击操作 (tap)")

        # 记录操作摘要
        print(f"\n🚀 操作摘要:")
        print(f"  操作类型: {action_type.upper()}")
        print(f"  目标元素: {element_id}")
        print(f"  屏幕位置: ({center_x}, {center_y})")
        if action_type == "text":
            print(f"  输入文本: '{text}'")

        # 执行操作
        print("\n" + "="*80)
        print("🔄 开始执行设备操作...")
        print(f"🔍 调用 screen_action.invoke()")
        print(f"  参数类型: {type(action_params)}")
        print(f"  参数内容: {json.dumps(action_params, indent=2, default=str, ensure_ascii=False)}")
        
        # 验证必要参数
        if action_type == "tap" and (action_params.get('x') is None or action_params.get('y') is None):
            print("❌ 错误: 点击操作需要 x 和 y 坐标")
            return False
        elif action_type == "swipe" and (action_params.get('start') is None or action_params.get('end') is None):
            print("❌ 错误: 滑动操作需要 start 和 end 坐标")
            return False
            
        try:
            # 记录开始时间
            action_start_time = time_module.time()
            
            # 执行操作
            print("\n" + "="*80)
            print("🔵 [DEBUG] 准备调用 screen_action.invoke()")
            print(f"🔵 [DEBUG] 参数类型: {type(action_params)}")
            print(f"🔵 [DEBUG] 参数内容: {json.dumps(action_params, indent=2, default=str, ensure_ascii=False)}")
            
            # 确保 screen_action 是可调用的
            if not callable(screen_action.invoke):
                print("❌ [DEBUG] screen_action.invoke 不是可调用对象")
                return False
            
            # 打印 screen_action 的详细信息
            print(f"\n🔵 [DEBUG] screen_action 对象信息:")
            print(f"  类型: {type(screen_action)}")
            print(f"  模块: {getattr(screen_action, '__module__', 'unknown')}")
            print(f"  名称: {getattr(screen_action, '__name__', 'unknown')}")
            print(f"  文档: {getattr(screen_action, '__doc__', 'No docstring')}")
            
            # 打印 screen_action.invoke 的详细信息
            print(f"\n🔵 [DEBUG] screen_action.invoke 方法信息:")
            print(f"  类型: {type(screen_action.invoke)}")
            print(f"  可调用: {callable(screen_action.invoke)}")
            
            # 记录调用前时间
            import time
            start_time = time.time()
            
            try:
                print("\n🔵 [DEBUG] 正在调用 screen_action.invoke...")
                result = screen_action.invoke(action_params)
                execution_time = (time.time() - start_time) * 1000  # 转换为毫秒
                print(f"✅ [DEBUG] screen_action.invoke 调用成功 (耗时: {execution_time:.2f}ms)")
                print(f"  返回结果类型: {type(result)}")
                print(f"  返回结果内容: {result}")
            except Exception as e:
                execution_time = (time.time() - start_time) * 1000
                print(f"❌ [DEBUG] screen_action.invoke 调用失败 (耗时: {execution_time:.2f}ms)")
                print(f"  错误类型: {type(e).__name__}")
                print(f"  错误信息: {str(e)}")
                print(f"  错误详情: {e}")
                print("  错误堆栈:")
                import traceback
                traceback.print_exc()
                return False
            
            # 计算执行时间
            execution_time_ms = (time_module.time() - action_start_time) * 1000
            
            print(f"\n✅ 操作执行完成")
            print(f"  返回值: {result} (类型: {type(result)})")
            print(f"  执行耗时: {execution_time_ms:.2f}ms")
            print(f"  调用后页面元素数量: {len(state['current_page'].get('elements_data', []))}")
            print("="*80 + "\n")

            # 解析操作结果
            print("\n🔍 解析操作结果...")
            if isinstance(result, str):
                try:
                    # 尝试解析JSON结果
                    result_json = json.loads(result)
                    print("  成功解析为JSON")
                    print(f"  解析后内容: {json.dumps(result_json, indent=2, ensure_ascii=False, default=str)}")
                    
                    status = result_json.get("status", "unknown").lower()
                    message = result_json.get("message", "")
                    
                    if status == "success":
                        print("\n🎉 操作执行成功!")
                        if message:
                            print(f"   返回信息: {message}")
                        return True
                    else:
                        error_msg = message if message else "未提供错误信息"
                        print(f"\n❌ 操作执行失败: {error_msg}")
                        print(f"   状态码: {status}")
                        return False
                        
                except json.JSONDecodeError:
                    print(f"\n⚠️ 操作结果不是有效的JSON")
                    print(f"  原始返回: {result}")
                    
                    # 检查是否包含错误信息
                    if any(err in result.lower() for err in ["error", "fail", "exception"]):
                        print("❌ 检测到可能的错误信息")
                        return False
                    else:
                        print("ℹ️ 将非JSON响应视为成功")
                        return True
                        
            elif result is None:
                print("\n⚠️ 操作返回了 None")
                print("ℹ️ 将None响应视为成功")
                return True
                
            else:
                print(f"\n⚠️ 操作返回了非字符串结果: {type(result).__name__}")
                print(f"  返回内容: {str(result)[:200]}{'...' if len(str(result)) > 200 else ''}")
                
                # 对于非字符串的返回值，检查是否有错误指示
                if hasattr(result, "get"):
                    error = result.get("error") or result.get("status") == "error"
                    if error:
                        print(f"❌ 检测到错误: {error}")
                        return False
                
                print("ℹ️ 将非字符串响应视为成功")
                return True
                
        except Exception as e:
            print(f"\n❌❌❌ 执行操作时发生异常 ❌❌❌")
            print(f"  错误类型: {type(e).__name__}")
            print(f"  错误信息: {str(e)}")
            print("\n堆栈跟踪:")
            traceback.print_exc()
            return False

    except Exception as e:
        import traceback
        print(f"\n❌❌❌ 执行元素操作时发生未捕获的异常 ❌❌❌")
        print(f"错误类型: {type(e).__name__}")
        print(f"错误信息: {str(e)}")
        print("\n堆栈跟踪:")
        traceback.print_exc()
        return False
    finally:
        print("="*80 + "\n")


def fallback_to_react(state: DeploymentState) -> DeploymentState:
    """
    Fall back to React mode when template execution fails

    Args:
        state: Execution state

    Returns:
        Updated execution state
    """
    print("🔄 Falling back to React mode execution...")
    task = state["task"]

    # Create action_agent for page operation decisions
    action_agent = create_react_agent(model, [screen_action])

    # Initialize React mode
    if not state["messages"]:
        # Set system prompt
        system_message = SystemMessage(
            content="""You are an intelligent smartphone operation assistant who will help users complete tasks on mobile devices.
You can help users by observing the screen and performing various operations (clicking, typing text, swiping, etc.).
Analyze the current screen content, determine the best next action, and use the appropriate tools to execute it.
Each step of the operation should move toward completing the user's goal task."""
        )

        state["messages"].append(system_message)

        # Add user task
        user_message = HumanMessage(
            content=f"I need to complete the following task on a mobile device: {task}"
        )
        state["messages"].append(user_message)

    # Capture current screen
    state = capture_and_parse_screen(state)
    if not state["current_page"]["screenshot"]:
        state["execution_status"] = "error"
        print("Unable to capture or parse screen")
        return state

    # Prepare screen information
    screenshot_path = state["current_page"]["screenshot"]
    elements_json_path = state["current_page"]["elements_json"]
    device = state["device"]
    device_size = get_device_size.invoke(device)

    # Load screenshot as base64
    with open(screenshot_path, "rb") as f:
        image_data = f.read()
        image_data_base64 = base64.b64encode(image_data).decode("utf-8")

    # Load element JSON data
    with open(elements_json_path, "r", encoding="utf-8") as f:
        elements_data = json.load(f)

    elements_text = json.dumps(elements_data, ensure_ascii=False, indent=2)

    # Build messages
    messages = [
        SystemMessage(
            content=f"""Below is the current page information and user intent. Please analyze comprehensively and recommend the next reasonable action (please complete only one step),
and complete it by calling tools. All tool calls must pass in device to specify the operating device. Only execute one tool call."""
        ),
        HumanMessage(
            content=f"The current device is: {device}, the device screen size is {device_size}. The user's current task intent is: {task}"
        ),
        HumanMessage(
            content="Below is the current page's parsed JSON data (where bbox is a relative value, please convert to actual operation position based on screen size):\n"
            + elements_text
        ),
        HumanMessage(
            content=[
                {"type": "text", "text": "Below is the screenshot data:"},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image_data_base64}"},
                },
            ],
        ),
    ]

    # Add these messages to state
    state["messages"].extend(messages)

    # Call action_agent for decision making and action execution
    action_result = action_agent.invoke({"messages": state["messages"][-4:]})

    # Parse results
    final_messages = action_result.get("messages", [])
    if final_messages:
        # Add AI reply to message history
        ai_message = final_messages[-1]
        state["messages"].append(ai_message)

        # Extract recommended action from final_message
        recommended_action = ai_message.content.strip()

        # Update execution status
        state["current_step"] += 1
        state["history"].append(
            {
                "step": state["current_step"],
                "screenshot": screenshot_path,
                "elements_json": elements_json_path,
                "action": "react_mode",
                "recommended_action": recommended_action,
                "status": "success",
            }
        )

        state["execution_status"] = "success"
        print(f"✓ React mode execution successful: {recommended_action}")
    else:
        error_msg = "React mode execution failed: No messages returned"
        print(f"❌ {error_msg}")

        # Update execution status
        state["history"].append(
            {
                "step": state["current_step"],
                "screenshot": screenshot_path,
                "elements_json": elements_json_path,
                "action": "react_mode",
                "status": "error",
                "error": error_msg,
            }
        )

        state["execution_status"] = "error"

    return state


def execute_task(
    state: DeploymentState, task: str, device: str, neo4j_db: Neo4jDatabase = None
) -> Dict[str, Any]:
    """
    Main function to execute a task

    Args:
        state: Initial state
        task: User task
        device: Device ID
        neo4j_db: Neo4j database connection (optional, uses global db by default)

    Returns:
        Execution result
    """
    print("\n" + "="*80)
    print("🚀 EXECUTE TASK - 开始执行任务")
    print(f"📝 任务描述: {task}")
    print(f"📱 目标设备: {device}")
    print("="*80 + "\n")

    if neo4j_db is None:
        neo4j_db = db  # Use global db if not provided

    try:
        from data.State import create_deployment_state

        # Create new state
        print("🔄 正在初始化任务状态...")
        state = create_deployment_state(task=task, device=device)
        print("✅ 任务状态初始化完成")

        # Use global db object
        neo4j_db = neo4j_db or db

        # Query database for all element nodes
        print("\n🔍 查询数据库获取所有元素节点...")
        all_elements = neo4j_db.get_all_actions()
        if not all_elements:
            print("⚠️ 数据库中未找到元素节点，回退到React模式")
            state = fallback_to_react(state)
            return {"status": state["execution_status"], "state": state}
        print(f"✅ 成功获取 {len(all_elements)} 个元素节点")

        # Query database for high-level actions related to the task
        print(f"\n🔍 查询与任务相关的高级操作...")
        high_level_actions = neo4j_db.get_high_level_actions_for_task(task)
        
        if high_level_actions:
            print(f"✅ 找到 {len(high_level_actions)} 个与任务相关的高级操作")
            
            # 打印高级操作详情
            for i, action in enumerate(high_level_actions, 1):
                print(f"\n🔹 高级操作 {i}/{len(high_level_actions)}:")
                print(f"   ID: {action.get('action_id', 'N/A')}")
                print(f"   描述: {action.get('description', 'N/A')}")
                print(f"   应用: {action.get('app_name', 'N/A')}")
                print(f"   包名: {action.get('package_name', 'N/A')}")
                
                # 打印操作步骤
                if 'action_sequence' in action and action['action_sequence']:
                    print(f"   \n   操作步骤 ({len(action['action_sequence'])} 步):")
                    for j, step in enumerate(action['action_sequence'], 1):
                        print(f"   {j}. 类型: {step.get('type', 'N/A')}")
                        print(f"      元素ID: {step.get('element_id', 'N/A')}")
                        print(f"      描述: {step.get('description', 'N/A')}")
                        if 'x' in step and 'y' in step:
                            print(f"      坐标: ({step.get('x')}, {step.get('y')})")
                        print()
                else:
                    print("   ❌ 没有找到操作步骤")
                    
                print("-" * 50)

            # Check for shortcut associations
            print("\n🔄 检查快捷方式关联...")
            shortcuts = check_shortcut_associations(state, high_level_actions)

            if shortcuts:
                print(f"✅ 找到 {len(shortcuts)} 个可能的快捷方式")

                # Evaluate shortcut execution conditions
                print("\n🔍 评估快捷方式执行条件...")
                valid_shortcuts = evaluate_shortcut_execution(state, shortcuts)

                if valid_shortcuts:
                    print(f"✅ 有 {len(valid_shortcuts)} 个快捷方式满足执行条件")

                    # Generate execution template
                    print("\n🔄 生成执行模板...")
                    execution_template = generate_execution_template(state, valid_shortcuts)

                    if execution_template:
                        print("✅ 执行模板生成成功")

                        # Sort shortcuts by priority
                        print("\n📊 根据优先级排序快捷方式...")
                        prioritized_shortcuts = prioritize_shortcuts(state, valid_shortcuts)
                        print(f"  已排序 {len(prioritized_shortcuts)} 个快捷方式")

                        # Execute high-level operation
                        print("\n🚀 开始执行高级操作...")
                        result = execute_high_level_action(
                            state, prioritized_shortcuts, execution_template
                        )

                        if result.get("status") == "success":
                            print(f"\n🎉 高级操作执行成功: {result.get('message', '')}")
                            state["execution_status"] = "success"
                            state["completed"] = True
                            return {"status": "success", "state": state}
                        else:
                            error_msg = result.get('message', '未知错误')
                            print(f"\n❌ 高级操作执行失败: {error_msg}")
                            # Fall back to React mode on failure
                            print("\n🔄 回退到React模式...")
                            state = fallback_to_react(state)
                            return {
                                "status": state["execution_status"],
                                "state": state,
                                "error": error_msg
                            }
            else:
                print("ℹ️ 未找到可用的快捷方式，尝试执行基本操作序列")
                
                # No shortcuts, try executing basic operation sequence
                for action in high_level_actions:
                    action_sequence = action.get("action_sequence", [])
                    if not action_sequence:
                        print("⚠️ 操作序列为空，跳过")
                        continue

                    print(f"\n📋 执行操作序列 (共 {len(action_sequence)} 步)")
                    
                    # Capture and parse screen
                    print("\n📸 捕获并解析屏幕...")
                    state = capture_and_parse_screen(state)
                    if not state["current_page"]["screenshot"]:
                        state["retry_count"] += 1
                        print(f"⚠️ 第 {state['retry_count']} 次尝试捕获/解析屏幕失败")
                        
                        if state["retry_count"] >= state["max_retries"]:
                            print(f"❌ 连续 {state['max_retries']} 次捕获/解析屏幕失败，回退到React模式")
                            state = fallback_to_react(state)
                            return {
                                "status": state["execution_status"],
                                "state": state,
                                "error": "Failed to capture/parse screen"
                            }
                        continue

                    # Reset retry count
                    state["retry_count"] = 0
                    print("✅ 屏幕捕获并解析成功")

                    # Match screen elements
                    print(f"\n🔍 匹配屏幕元素 (步骤 {state['current_step'] + 1}/{len(action_sequence)})...")
                    element_matches = match_screen_elements(state, action_sequence)
                    
                    if not element_matches:
                        state["retry_count"] += 1
                        print(f"⚠️ 第 {state['retry_count']} 次尝试匹配元素失败")
                        
                        if state["retry_count"] >= state["max_retries"]:
                            print(f"❌ 连续 {state['max_retries']} 次未找到匹配元素，回退到React模式")
                            state = fallback_to_react(state)
                            return {
                                "status": state["execution_status"],
                                "state": state,
                                "error": "No matching elements found"
                            }
                        continue

                    # Reset retry count
                    state["retry_count"] = 0
                    print(f"✅ 找到 {len(element_matches)} 个匹配的元素")

                    # Execute element action
                    best_match = element_matches[0]
                    current_step = state["current_step"]
                    
                    print("\n" + "="*80)
                    print(f"🔄 准备执行步骤 {current_step + 1}/{len(action_sequence)} - 元素操作")
                    print("🔍 匹配到的元素详情:")
                    print(f"  元素ID: {best_match.get('element_id', 'N/A')}")
                    print(f"  屏幕元素ID: {best_match.get('screen_element_id', 'N/A')}")
                    print(f"  操作类型: {best_match.get('action_type', 'tap')}")
                    print(f"  元素位置: {best_match.get('position', 'N/A')}")
                    print(f"  元素内容: {best_match.get('content', 'N/A')}")
                    
                    # 添加操作前延迟
                    action_delay = state.get("action_delay", 1.0)
                    print(f"⏳ 等待 {action_delay} 秒后执行操作...")
                    time.sleep(action_delay)
                    
                    print("\n" + "="*50)
                    print("🔵 准备执行元素操作")
                    print(f"🔵 当前步骤: {current_step + 1}/{len(action_sequence)}")
                    print(f"🔵 匹配到的元素: {json.dumps(best_match, indent=2, ensure_ascii=False, default=str)}")
                    
                    # 检查页面元素数据
                    if "current_page" not in state or "elements_data" not in state["current_page"]:
                        print("❌ 错误: 页面元素数据缺失")
                        return state
                        
                    elements_data = state["current_page"]["elements_data"]
                    print(f"🔵 当前页面元素数量: {len(elements_data)}")
                    
                    # 验证屏幕元素ID
                    screen_element_id = best_match.get("screen_element_id")
                    if not isinstance(screen_element_id, int) or screen_element_id < 0 or screen_element_id >= len(elements_data):
                        print(f"❌ 错误: 无效的屏幕元素ID: {screen_element_id}")
                        print(f"❌ 有效范围: 0-{len(elements_data)-1}")
                        return state
                    
                    # 获取目标元素信息
                    target_element = elements_data[screen_element_id]
                    print(f"🔵 目标元素信息: {json.dumps(target_element, indent=2, ensure_ascii=False, default=str)}")
                    
                    # 记录调用前状态
                    prev_elements_count = len(elements_data)
                    print(f"🔵 调用前页面元素数量: {prev_elements_count}")
                    
                    # 记录开始时间
                    start_time = time_module.time()
                    
                    print("\n" + "="*50)
                    print("🔄 开始执行元素操作...")
                    print(f"🔍 调用 execute_element_action")
                    print(f"  参数 - state.keys(): {list(state.keys())}")
                    print(f"  参数 - best_match: {json.dumps(best_match, indent=2, ensure_ascii=False, default=str)}")
                    
                    # 执行元素操作
                    success = execute_element_action(state, best_match)
                    
                    # 记录耗时
                    execution_time = time_module.time() - start_time
                    
                    print(f"\n✅ execute_element_action 执行完成")
                    print(f"  返回值: {success} (类型: {type(success)})")
                    print(f"  执行耗时: {execution_time:.2f}秒")
                    print(f"  调用后页面元素数量: {len(state['current_page'].get('elements_data', []))}")
                    print("="*80 + "\n")

                    if success:
                        print(f"\n✅ 步骤 {current_step + 1} 执行成功")
                        
                        # Update history before incrementing step
                        history_entry = {
                            "step": current_step + 1,
                            "screenshot": state["current_page"]["screenshot"],
                            "elements_json": state["current_page"]["elements_json"],
                            "action": best_match.get("action_type", "tap"),
                            "element_id": best_match.get("element_id", ""),
                            "screen_element_id": best_match.get("screen_element_id", -1),
                            "status": "success",
                            "timestamp": time.time()
                        }
                        state["history"].append(history_entry)
                        print(f"📝 已更新历史记录 (总历史记录数: {len(state['history'])})")

                        # Update current step after successful execution
                        state["current_step"] = current_step + 1
                        state["retry_count"] = 0  # Reset retry count on success
                        print(f"📊 当前进度: {state['current_step']}/{len(action_sequence)}")

                        # Add delay between actions
                        time.sleep(1)  # 1 second delay between actions

                        # Check if all steps are completed
                        if state["current_step"] >= len(action_sequence):
                            print("\n🎉 所有步骤执行完成！")
                            state["execution_status"] = "success"
                            state["completed"] = True
                            return {
                                "status": "success",
                                "state": state,
                                "message": "All steps completed successfully"
                            }
                            
                        # 继续执行下一步
                        continue
                    else:
                        current_step = state["current_step"]
                        print(f"\n❌ 步骤 {current_step + 1} 执行失败")

                        # Update history
                        history_entry = {
                            "step": current_step + 1,
                            "screenshot": state["current_page"]["screenshot"],
                            "elements_json": state["current_page"]["elements_json"],
                            "action": best_match.get("action_type", "tap"),
                            "element_id": best_match.get("element_id", ""),
                            "screen_element_id": best_match.get("screen_element_id", -1),
                            "status": "error",
                            "retry_count": state.get("retry_count", 0) + 1,
                            "timestamp": time.time()
                        }
                        state["history"].append(history_entry)
                        print(f"📝 已更新历史记录 (总历史记录数: {len(state['history'])})")

                        # Increment retry count
                        state["retry_count"] = state.get("retry_count", 0) + 1
                        print(f"🔄 重试计数: {state['retry_count']}/{state['max_retries']}")
                        
                        if state["retry_count"] >= state["max_retries"]:
                            print(f"\n❌ 操作连续失败 {state['max_retries']} 次，回退到React模式")
                            state = fallback_to_react(state)
                            return {
                                "status": state["execution_status"],
                                "state": state,
                                "error": f"Operation failed after {state['max_retries']} retries"
                            }
                        
                        # 添加重试延迟，随着重试次数增加而增加
                        retry_delay = min(5, 1 * (state["retry_count"] + 1))  # 最大5秒
                        print(f"⏳ 等待 {retry_delay} 秒后重试...")
                        time.sleep(retry_delay)
                        
                        # 重新捕获屏幕以获取最新状态
                        print("🔄 重新捕获屏幕以获取最新状态...")
                        state = capture_and_parse_screen(state)
                        if not state["current_page"]["screenshot"]:
                            print("❌ 重新捕获屏幕失败，无法继续重试")
                            state = fallback_to_react(state)
                            return {
                                "status": state["execution_status"],
                                "state": state,
                                "error": "Failed to recapture screen for retry"
                            }
        else:
            print("❌ 未找到与任务匹配的高级操作，回退到React模式")
            state = fallback_to_react(state)
            return {
                "status": state["execution_status"],
                "state": state,
                "error": "No matching high-level actions found"
            }

        # If all above methods fail, fall back to React mode
        print("\n⚠️ 无法使用高级操作完成任务，回退到基本操作空间")
        state = fallback_to_react(state)
        return {
            "status": state["execution_status"],
            "state": state,
            "error": "All execution methods failed"
        }
        
    except Exception as e:
        import traceback
        print("\n" + "❌" * 20 + " 未捕获的异常 " + "❌" * 20)
        print(f"错误类型: {type(e).__name__}")
        print(f"错误信息: {str(e)}")
        print("\n堆栈跟踪:")
        traceback.print_exc()
        print("❌" * 50 + "\n")
        
        # 确保状态被正确更新
        state["execution_status"] = "error"
        state["error"] = f"Unhandled exception: {str(e)}"
        
        # 尝试回退到React模式
        try:
            state = fallback_to_react(state)
        except Exception as fallback_error:
            print(f"❌ 回退到React模式时出错: {str(fallback_error)}")
        
        return {
            "status": "error",
            "state": state,
            "error": f"Unhandled exception: {str(e)}",
            "traceback": traceback.format_exc()
        }


def run_task(task: str, device: str = "emulator-5554") -> Dict[str, Any]:
    """
    Execute a single task

    Args:
        task: User task description
        device: Device ID

    Returns:
        Execution result
    """
    print(f"🚀 Starting task execution: {task}")

    try:
        # Initialize state using create_deployment_state function
        from data.State import create_deployment_state

        state = create_deployment_state(
            task=task,
            device=device,
            max_retries=3,
        )

        # Execute task using LangGraph workflow
        workflow = build_workflow()
        app = workflow.compile()
        result = app.invoke(state)

        # Display final screenshot if execution was successful
        if (
            result["execution_status"] == "success"
            and result["current_page"]["screenshot"]
        ):
            try:
                from PIL import Image

                img = Image.open(result["current_page"]["screenshot"])
                img.show()
            except Exception as e:
                print(f"Unable to display final screenshot: {str(e)}")

        return {
            "status": result["execution_status"],
            "message": "Task execution completed",
            "steps_completed": result["current_step"],
            "total_steps": result["total_steps"],
        }

    except Exception as e:
        print(f"❌ Error executing task: {str(e)}")
        return {
            "status": "error",
            "message": f"Error executing task: {str(e)}",
            "error": str(e),
        }


def check_shortcut_associations(
    state: DeploymentState, high_level_actions: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Check if high-level actions are associated with shortcuts

    Args:
        state: Execution state
        high_level_actions: List of high-level actions

    Returns:
        List of associated shortcuts
    """
    print("🔍 Checking high-level action shortcut associations...")
    shortcuts = []

    for action in high_level_actions:
        action_id = action.get("action_id")
        if not action_id:
            continue

        # Query database for shortcuts associated with high-level action
        associated_shortcuts = state["neo4j_db"].get_shortcuts_for_action(action_id)
        if associated_shortcuts:
            for shortcut in associated_shortcuts:
                shortcuts.append(
                    {
                        "shortcut_id": shortcut.get("shortcut_id"),
                        "name": shortcut.get("name"),
                        "description": shortcut.get("description"),
                        "action_id": action_id,
                        "action_name": action.get("name"),
                        "action_sequence": action.get("action_sequence", []),
                        "conditions": shortcut.get("conditions", {}),
                        "priority": shortcut.get("priority", 0),
                        "page_flow": shortcut.get("page_flow", []),
                    }
                )

    if shortcuts:
        print(f"✓ Found {len(shortcuts)} associated shortcuts")
    else:
        print("⚠️ No associated shortcuts found")

    return shortcuts


def evaluate_shortcut_execution(
    state: DeploymentState, shortcuts: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Evaluate if shortcuts meet execution conditions

    Args:
        state: Execution state
        shortcuts: List of shortcuts

    Returns:
        List of shortcuts that meet execution conditions
    """
    print("🧠 Evaluating shortcut execution conditions...")

    if not shortcuts:
        print("⚠️ No shortcuts to evaluate")
        return []

    # Prepare current screen information
    screen_desc = ""
    if state["current_page"]["elements_data"]:
        screen_desc = "Current screen contains the following elements:\n"
        for i, element in enumerate(state["current_page"]["elements_data"]):
            element_type = element.get("type", "Unknown type")
            element_content = element.get("content", "")
            screen_desc += f"{i+1}. Type: {element_type}, Content: {element_content}\n"

    # Prepare task information
    task_desc = state["task"]

    # Create evaluation prompt
    eval_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are a smartphone operation assistant responsible for evaluating whether the current scenario meets the conditions for executing shortcuts.
Analyze the current screen information, user task, and shortcut execution conditions to determine which shortcuts can be executed.
Only execute shortcuts when their conditions match the current scenario.""",
            ),
            (
                "human",
                """User task: {task}

Current screen information:
{screen_info}

Available shortcuts:
{shortcuts_info}

Please evaluate if each shortcut meets execution conditions, return in JSON format:
{{
  "valid_shortcuts": [
    {{
      "shortcut_id": "ID of shortcut meeting conditions",
      "reason": "Reason for meeting conditions",
      "confidence": "Confidence level between 0.0-1.0"
    }},
    ...
  ]
}}

If no shortcuts meet conditions, return an empty list.
""",
            ),
        ]
    )

    # Prepare shortcuts information
    shortcuts_info = ""
    for i, shortcut in enumerate(shortcuts):
        shortcuts_info += f"{i+1}. ID: {shortcut.get('shortcut_id')}\n"
        shortcuts_info += f"   Name: {shortcut.get('name')}\n"
        shortcuts_info += (
            f"   Description: {shortcut.get('description', 'No description')}\n"
        )

        # Add conditions information
        conditions = shortcut.get("conditions", {})
        if conditions:
            shortcuts_info += "   Execution conditions:\n"
            if isinstance(conditions, dict):
                for k, v in conditions.items():
                    shortcuts_info += f"     - {k}: {v}\n"
            elif isinstance(conditions, str):
                shortcuts_info += f"     - {conditions}\n"

        shortcuts_info += "\n"

    # Call LLM for evaluation
    try:
        # Prepare input
        eval_input = {
            "task": task_desc,
            "screen_info": screen_desc,
            "shortcuts_info": shortcuts_info,
        }

        # Create parser
        parser = JsonOutputParser()

        # Build chain
        eval_chain = eval_prompt | model | parser

        # Execute evaluation
        result = eval_chain.invoke(eval_input)

        # Parse results
        valid_shortcuts = result.get("valid_shortcuts", [])

        if valid_shortcuts:
            # Find corresponding complete shortcut information
            valid_shortcut_objects = []
            for valid in valid_shortcuts:
                shortcut_id = valid.get("shortcut_id")
                for shortcut in shortcuts:
                    if shortcut.get("shortcut_id") == shortcut_id:
                        # Add evaluation information
                        shortcut_copy = shortcut.copy()
                        shortcut_copy["evaluation"] = {
                            "reason": valid.get("reason", ""),
                            "confidence": valid.get("confidence", 0.0),
                        }
                        valid_shortcut_objects.append(shortcut_copy)
                        break

            print(
                f"✓ Found {len(valid_shortcut_objects)} shortcuts meeting execution conditions"
            )
            return valid_shortcut_objects
        else:
            print("⚠️ No shortcuts meet execution conditions")
            return []

    except Exception as e:
        print(f"❌ Error evaluating shortcut execution conditions: {str(e)}")
        return []


def generate_execution_template(
    state: DeploymentState, shortcuts: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Generate execution template based on shortcuts

    Args:
        state: Execution state
        shortcuts: List of shortcuts that meet execution conditions

    Returns:
        Execution template with operation steps and parameters
    """
    print("📝 Generating execution template...")

    if not shortcuts:
        print("⚠️ No available shortcuts, cannot generate execution template")
        return {}

    # Select shortcut with highest confidence
    selected_shortcut = max(
        shortcuts, key=lambda x: x.get("evaluation", {}).get("confidence", 0)
    )

    # Get device dimensions
    device_size = get_device_size.invoke(state["device"])
    if isinstance(device_size, str):
        device_size = {"width": 1080, "height": 1920}

    # Prepare current screen information
    screen_elements = state["current_page"]["elements_data"]
    screen_elements_json = json.dumps(screen_elements, ensure_ascii=False, indent=2)

    # Create template generation prompt
    template_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are a smartphone operation assistant responsible for generating detailed execution templates based on shortcuts and current screen state.
The execution template should include all steps needed to complete the operation, with each step specifying the action type, target element, and necessary parameters.
Available action types include: tap, text (input text), swipe, long_press, back.
Ensure the generated template can accurately execute the operations described in the shortcut.""",
            ),
            (
                "human",
                """Shortcut information:
{shortcut_info}

Current screen elements:
{screen_elements}

Device size: {device_size}

Please generate a detailed template for executing this shortcut, return in JSON format:
{
  "steps": [
    {
      "action_type": "action type(tap/text/swipe/long_press/back)",
      "target_element_id": target element ID(number),
      "parameters": {
        // Add appropriate parameters based on action type
        // e.g., text operation needs "text" parameter
        // swipe operation needs "direction" and "distance" parameters
      }
    },
    // more steps...
  ]
}

Ensure each step has a clear operation target and necessary parameters. If the operation doesn't need a target element (like back operation), you can omit target_element_id.
""",
            ),
        ]
    )

    # Prepare shortcut information
    shortcut_info = f"ID: {selected_shortcut.get('shortcut_id')}\n"
    shortcut_info += f"Name: {selected_shortcut.get('name')}\n"
    shortcut_info += (
        f"Description: {selected_shortcut.get('description', 'No description')}\n"
    )

    # Add action sequence information
    action_sequence = selected_shortcut.get("action_sequence", [])
    if action_sequence:
        shortcut_info += "Action sequence:\n"
        if isinstance(action_sequence, list):
            for i, action in enumerate(action_sequence):
                shortcut_info += f"  {i+1}. {json.dumps(action, ensure_ascii=False)}\n"
        elif isinstance(action_sequence, str):
            shortcut_info += f"  {action_sequence}\n"

    # Add evaluation information
    evaluation = selected_shortcut.get("evaluation", {})
    if evaluation:
        shortcut_info += f"Execution reason: {evaluation.get('reason', 'None')}\n"
        shortcut_info += f"Confidence: {evaluation.get('confidence', 0.0)}\n"

    # Call LLM to generate template
    try:
        # Prepare input
        template_input = {
            "shortcut_info": shortcut_info,
            "screen_elements": screen_elements_json,
            "device_size": json.dumps(device_size, ensure_ascii=False),
        }

        # Create parser
        parser = JsonOutputParser()

        # Build chain
        template_chain = template_prompt | model | parser

        # Execute generation
        result = template_chain.invoke(template_input)

        # Validate result
        if (
            "steps" in result
            and isinstance(result["steps"], list)
            and len(result["steps"]) > 0
        ):
            print(
                f"✓ Successfully generated execution template with {len(result['steps'])} steps"
            )
            return result
        else:
            print("❌ Generated execution template is invalid")
            return {}

    except Exception as e:
        print(f"❌ Error generating execution template: {str(e)}")
        return []


def prioritize_shortcuts(
    state: Dict[str, Any], shortcuts: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Prioritize shortcuts based on page flow

    Args:
        state: Execution state
        shortcuts: List of shortcuts

    Returns:
        Prioritized list of shortcuts
    """
    if not shortcuts or len(shortcuts) <= 1:
        return shortcuts

    try:
        # Get page flow information from database
        page_flow = state["neo4j_db"].get_page_flow()

        if not page_flow:
            print("⚠️ No page flow information found, using default sorting")
            # Default sort by match score
            return sorted(
                shortcuts,
                key=lambda x: x["element_match"].get("match_score", 0),
                reverse=True,
            )

        # Assign priority to each shortcut based on page flow position
        prioritized = []
        for shortcut in shortcuts:
            # Find shortcut position in page flow
            position = -1
            shortcut_id = shortcut["shortcut_id"]

            for idx, flow_node in enumerate(page_flow):
                if flow_node.get("shortcut_id") == shortcut_id:
                    position = idx
                    break

            prioritized.append(
                {
                    "shortcut": shortcut,
                    "flow_position": position,
                    "match_score": shortcut["element_match"].get("match_score", 0),
                }
            )

        # First sort by flow position, unknown positions (-1) at the end
        # For same positions, sort by match score
        prioritized.sort(
            key=lambda x: (
                x["flow_position"] if x["flow_position"] >= 0 else float("inf"),
                -x["match_score"],
            )
        )

        return [item["shortcut"] for item in prioritized]

    except Exception as e:
        print(f"⚠️ Shortcut prioritization failed: {str(e)}")
        # Return original list on error
        return shortcuts


def execute_high_level_action(
    state: DeploymentState,
    shortcuts: List[Dict[str, Any]],
    execution_template: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Execute high-level operations

    Args:
        state: Execution state
        shortcuts: List of shortcuts meeting execution conditions
        execution_template: Execution template

    Returns:
        Execution result
    """
    print("\n" + "="*80)
    print("🚀 EXECUTE HIGH LEVEL ACTION - 开始执行高级操作")
    print(f"  - Template steps: {len(execution_template.get('steps', []))}")
    print(f"  - Shortcuts count: {len(shortcuts)}")
    if shortcuts:
        print(f"  - First shortcut: {shortcuts[0].get('name', 'Unknown')}")
    print("="*80 + "\n")

    if not execution_template or "steps" not in execution_template:
        error_msg = "❌ Invalid execution template: missing steps"
        print(error_msg)
        return {"status": "error", "message": error_msg}
        
    steps = execution_template.get("steps", [])
    print(f"📋 Found {len(steps)} steps in execution template")

    steps = execution_template["steps"]
    if not steps or not isinstance(steps, list):
        print("❌ No valid steps in execution template")
        return {"status": "error", "message": "No valid steps in execution template"}

    # Initialize execution state
    state["current_step"] = 0
    state["total_steps"] = len(steps)
    state["execution_status"] = "running"
    state["history"] = []

    # Record shortcut execution start
    shortcut_names = ", ".join([s.get("name", "Unnamed shortcut") for s in shortcuts])
    print(f"Starting execution of shortcuts: {shortcut_names}")
    print(f"Total steps: {state['total_steps']}")

    # Execute each step
    while state["current_step"] < state["total_steps"]:
        current_step_idx = state["current_step"]
        step = steps[current_step_idx]
        
        print("\n" + "-"*60)
        print(f"🔄 EXECUTING STEP {current_step_idx + 1}/{state['total_steps']}")
        print(f"  - Step type: {step.get('action_type', 'unknown')}")
        print(f"  - Element ID: {step.get('element_id', 'N/A')}")
        print(f"  - Parameters: {json.dumps(step.get('parameters', {}), ensure_ascii=False, indent=2)}")
        print("-"*60 + "\n")

        # Capture and parse current screen
        state = capture_and_parse_screen(state)
        if not state["current_page"]["screenshot"]:
            state["retry_count"] += 1
            if state["retry_count"] >= state["max_retries"]:
                print(
                    f"❌ Failed to capture or parse screen {state['max_retries']} times in a row"
                )
                return {
                    "status": "error",
                    "message": "Unable to capture or parse screen",
                }

            # Wait a second before retrying
            time.sleep(1)
            continue

        # Reset retry counter
        state["retry_count"] = 0

        # Get action type and parameters
        action_type = step.get("action_type", "tap")
        target_element_id = step.get("target_element_id")
        parameters = step.get("parameters", {})

        # Special handling for back operation
        if action_type == "back":
            print("\n🔙 EXECUTING BACK OPERATION")
            print(f"  - Current step: {current_step_idx + 1}/{state['total_steps']}")
            print(f"  - Action type: {action_type}")
            result = screen_action.invoke({"device": state["device"], "action": "back"})

            # Record history
            state["history"].append(
                {
                    "step": current_step_idx,
                    "screenshot": state["current_page"]["screenshot"],
                    "elements_json": state["current_page"]["elements_json"],
                    "action": "back",
                    "status": "success",
                }
            )

            # Move to next step
            state["current_step"] += 1
            time.sleep(1)  # Wait for operation to take effect
            continue

        # For operations requiring target element
        print("\n🎯 TARGET ELEMENT OPERATION")
        print(f"  - Target element ID: {target_element_id}")
        print(f"  - Action type: {action_type}")
        print(f"  - Parameters: {json.dumps(parameters, ensure_ascii=False)}")
        
        if target_element_id is None:
            print("❌ Operation missing target element ID")
            return {
                "status": "error",
                "message": f"Step {current_step_idx + 1} missing target element ID",
            }

        # Check if target element exists
        screen_elements = state["current_page"]["elements_data"]
        print(f"  - Total elements on screen: {len(screen_elements) if screen_elements else 0}")
        
        if target_element_id < 0 or target_element_id >= len(screen_elements):
            print(f"❌ Invalid target element ID: {target_element_id}")
            if screen_elements:
                print("  - Available element IDs: ", 
                      ", ".join(str(i) for i in range(min(10, len(screen_elements)))) + 
                      ("..." if len(screen_elements) > 10 else ""))
            return {
                "status": "error",
                "message": f"Invalid target element ID for step {current_step_idx + 1}",
            }

        # Get element position and details
        element = screen_elements[target_element_id]
        bbox = element.get("bbox", [0, 0, 0, 0])
        
        print("\n📌 ELEMENT DETAILS:")
        print(f"  - Element ID: {target_element_id}")
        print(f"  - Bounding box: {bbox}")
        print(f"  - Text: {element.get('text', 'N/A')}")
        print(f"  - Resource ID: {element.get('resource-id', 'N/A')}")
        print(f"  - Class: {element.get('class', 'N/A')}")
        print(f"  - Package: {element.get('package', 'N/A')}")
        print(f"  - Content-desc: {element.get('content-desc', 'N/A')}")
        print(f"  - Clickable: {element.get('clickable', 'N/A')}")
        print(f"  - Visible: {element.get('visible', 'N/A')}")

        # Get device size and calculate center point
        device_size = get_device_size.invoke(state["device"])
        if isinstance(device_size, str):
            device_size = {"width": 1080, "height": 1920}
            print("⚠️ Using default device size (1080x1920)")
        else:
            print(f"  - Device size: {device_size['width']}x{device_size['height']}")

        # Calculate click position (relative to screen size)
        center_x = int((bbox[0] + bbox[2]) / 2 * device_size["width"])
        center_y = int((bbox[1] + bbox[3]) / 2 * device_size["height"])
        
        print(f"  - Click position (x,y): ({center_x}, {center_y})")

        # Prepare operation parameters
        action_params = {
            "device": state["device"],
            "action": action_type,
            "x": center_x,
            "y": center_y,
        }

        # Add specific parameters based on action type
        if action_type == "text":
            action_params["input_str"] = parameters.get("text", "")
        elif action_type == "long_press":
            action_params["duration"] = parameters.get("duration", 1000)
        elif action_type == "swipe":
            action_params["direction"] = parameters.get("direction", "up")
            action_params["dist"] = parameters.get("distance", "medium")

        # Execute operation
        print("\n🚀 EXECUTING ACTION:")
        print(f"  - Action: {action_type.upper()}")
        print(f"  - Position: ({center_x}, {center_y})")
        if action_type == "text":
            print(f"  - Input text: {action_params.get('input_str', '')}")
        elif action_type == "long_press":
            print(f"  - Duration: {action_params.get('duration', 1000)}ms")
        elif action_type == "swipe":
            print(f"  - Direction: {action_params.get('direction', 'up')}")
            print(f"  - Distance: {action_params.get('dist', 'medium')}")
            
        print(f"  - Action params: {json.dumps(action_params, default=str)}")
        print("-" * 60)

        result = screen_action.invoke(action_params)

        # Parse operation result
        success = False
        if isinstance(result, str):
            try:
                result_json = json.loads(result)
                if result_json.get("status") == "success":
                    success = True
            except:
                pass

        if success:
            print(f"✓ Step {current_step_idx + 1} executed successfully")

            # Record history
            state["history"].append(
                {
                    "step": current_step_idx,
                    "screenshot": state["current_page"]["screenshot"],
                    "elements_json": state["current_page"]["elements_json"],
                    "action": action_type,
                    "target_element_id": target_element_id,
                    "parameters": parameters,
                    "status": "success",
                }
            )

            # Move to next step
            state["current_step"] += 1
            state["retry_count"] = 0

            # Wait for operation to take effect
            time.sleep(1.5)
        else:
            print(f"❌ Step {current_step_idx + 1} execution failed")

            # Record history
            state["history"].append(
                {
                    "step": current_step_idx,
                    "screenshot": state["current_page"]["screenshot"],
                    "elements_json": state["current_page"]["elements_json"],
                    "action": action_type,
                    "target_element_id": target_element_id,
                    "parameters": parameters,
                    "status": "error",
                }
            )

            # Increment retry counter
            state["retry_count"] += 1
            if state["retry_count"] >= state["max_retries"]:
                print(f"❌ Operation failed {state['max_retries']} times in a row")
                return {
                    "status": "error",
                    "message": f"Step {current_step_idx + 1} execution failed",
                }

            # Wait a second before retrying
            time.sleep(1)

    # Capture final screen
    state = capture_and_parse_screen(state)

    # Execution complete
    print(
        f"\n✨ High-level operation execution complete! Completed {state['current_step']} operations"
    )
    state["execution_status"] = "success"
    state["completed"] = True

    return {
        "status": "success",
        "message": "Successfully executed high-level operations",
        "steps_completed": state["current_step"],
        "total_steps": state["total_steps"],
        "final_screenshot": state["current_page"]["screenshot"],
        "execution_history": state["history"],
    }


def capture_screen_node(state: DeploymentState) -> DeploymentState:
    print("📸 Capturing and parsing current screen...")

    state_dict = dict(state)
    updated_state = capture_and_parse_screen(state_dict)

    # Update state
    for key, value in updated_state.items():
        if key in state:
            state[key] = value

    if not state["current_page"]["screenshot"]:
        state["should_fallback"] = True
        print("❌ Unable to capture screen, marking for fallback")
    else:
        print("✓ Screen captured successfully")

    return state


def match_elements_node(state: DeploymentState) -> DeploymentState:
    """
    Match current screen elements using visual embeddings
    """
    print("🔍 Matching current screen elements using visual embeddings...")

    # Get all element nodes from database - using correct method name
    all_elements = db.get_all_actions()
    if not all_elements:
        print("⚠️ No element nodes in database, marking for fallback")
        state["should_fallback"] = True
        return state

    # Build action sequence with all elements from database
    action_sequence = []
    for element in all_elements:
        # Ensure element has element_id field
        if "element_id" in element:
            action_sequence.append(
                {
                    "element_id": element["element_id"],
                    "atomic_action": "tap",  # Default action
                    "action_params": {},
                }
            )
        else:
            # If no element_id, try using other possible ID fields
            element_id = (
                element.get("id") or element.get("node_id") or str(hash(str(element)))
            )
            print(
                f"⚠️ Element missing element_id field, using alternative ID: {element_id}"
            )
            action_sequence.append(
                {
                    "element_id": element_id,
                    "atomic_action": "tap",  # Default action
                    "action_params": {},
                }
            )

    # Call match_screen_elements function
    state_dict = dict(state)
    print("\n🔍 Starting visual element matching...")
    print(f"  - Elements to match: {len(action_sequence)}")
    if action_sequence:
        print(f"  - First element to match: {action_sequence[0].get('element_id', 'Unknown')}")
    
    element_matches = match_screen_elements(state_dict, action_sequence)
    state["matched_elements"] = element_matches
    
    # Debug: Print matched elements
    if element_matches:
        print("\n✅ Visual matching results:")
        for i, match in enumerate(element_matches, 1):
            print(f"  {i}. Element ID: {match.get('screen_element_id')}")
            print(f"     Action: {match.get('action_type')}")
            print(f"     Score: {match.get('match_score'):.4f}")
            if 'template_element' in match:
                print(f"     Template: {match['template_element'].get('id', 'N/A')}")
    else:
        print("\n❌ No elements matched via visual matching")

    if not state["matched_elements"]:
        print(
            "⚠️ No matching screen elements found, trying high-level task matching first"
        )

        # Try matching task to high-level actions
        is_matched, matched_action = match_task_to_action(state_dict, state["task"])

        if is_matched and matched_action:
            print(
                f"✓ Task matched to high-level action: {matched_action.get('name', 'Unknown')}"
            )
            # Save current executing high-level action
            state["current_action"] = matched_action

            # Get element sequence
            element_sequence = matched_action.get("element_sequence", [])
            if isinstance(element_sequence, str):
                try:
                    element_sequence = json.loads(element_sequence)
                except:
                    print(f"❌ Failed to parse element sequence")
                    state["should_fallback"] = True
                    return state

            if not element_sequence or not isinstance(element_sequence, list):
                print(f"❌ Element sequence is empty or incorrectly formatted")
                state["should_fallback"] = True
                return state

            # Update step information in state
            state["current_step"] = 0
            state["total_steps"] = len(element_sequence)

            # Recapture screen and match elements
            updated_state = capture_and_parse_screen(state_dict)
            for key, value in updated_state.items():
                if key in state:
                    state[key] = value

            element_matches = match_screen_elements(state_dict, element_sequence)
            state["matched_elements"] = element_matches

            if not element_matches:
                print(
                    "❌ Still no matching screen elements found, marking for fallback"
                )
                state["should_fallback"] = True
        else:
            print("❌ No matching high-level actions found, marking for fallback")
            state["should_fallback"] = True
    else:
        print(f"✓ Found {len(state['matched_elements'])} matching elements")

    return state


def check_shortcuts_node(state: DeploymentState) -> DeploymentState:
    """
    Check element associations with shortcuts
    """
    print("\n🔍 Checking element associations with shortcuts...")
    
    if state.get('matched_elements'):
        print(f"  Found {len(state['matched_elements'])} matched elements")
        for i, match in enumerate(state['matched_elements'], 1):
            element_id = match.get('screen_element_id')
            element = next((e for e in state.get('current_page', {}).get('elements_data', []) 
                          if str(e.get('ID')) == str(element_id)), None)
            print(f"\n  Match {i}:")
            print(f"    - Element ID: {element_id}")
            print(f"    - Match score: {match.get('match_score'):.4f}")
            print(f"    - Action type: {match.get('action_type')}")
            if element:
                print(f"    - Element details:")
                for k, v in element.items():
                    if k not in ['screenshot', 'screenshot_path']:  # Skip binary data
                        print(f"      - {k}: {v}")
    else:
        print("  No matched elements found in state")
        
    # Print current task information
    task = state.get('current_task', {})
    if task:
        print("\n📋 Current task:")
        print(f"  - Task ID: {task.get('id')}")
        print(f"  - Description: {task.get('description')}")
        print(f"  - Status: {task.get('status')}")

    if not state["matched_elements"]:
        print("⚠️ No matched elements, cannot check shortcut associations")
        state["should_fallback"] = True
        return state

    # Call check_shortcut_associations function
    state_dict = dict(state)
    associated_shortcuts = check_shortcut_associations(
        state_dict, state["matched_elements"]
    )
    state["associated_shortcuts"] = associated_shortcuts

    if state["associated_shortcuts"]:
        print(f"✓ Found {len(state['associated_shortcuts'])} associated shortcut nodes")

        # Priority sorting
        prioritized_shortcuts = prioritize_shortcuts(state_dict, associated_shortcuts)
        state["associated_shortcuts"] = prioritized_shortcuts
    else:
        print("📝 No associated shortcut nodes found")

    return state


def shortcut_evaluation_node(state: DeploymentState) -> DeploymentState:
    """
    Evaluate whether to execute shortcut operations
    """
    print("🧠 Evaluating whether to execute shortcut operations...")

    if not state["associated_shortcuts"]:
        print("⚠️ No associated shortcuts, skipping evaluation")
        return state

    # Call evaluate_shortcut_execution function
    state_dict = dict(state)
    execution_decision = evaluate_shortcut_execution(
        state_dict, state["associated_shortcuts"], state["task"]
    )

    state["should_execute_shortcut"] = execution_decision.get("should_execute", False)
    if "shortcut" in execution_decision:
        state["current_shortcut"] = execution_decision["shortcut"]

    if state["should_execute_shortcut"]:
        print(
            f"✓ Decided to execute shortcut: {state['current_shortcut'].get('name', 'Unknown')}"
        )
        print(f"  Reason: {execution_decision.get('reason', '')}")
    else:
        print(
            f"⚠️ Decided not to execute shortcut operations: {execution_decision.get('reason', '')}"
        )

    return state


def generate_template_node(state: DeploymentState) -> DeploymentState:
    """
    Generate execution template
    """
    print("📝 Generating execution template...")

    if not state["should_execute_shortcut"] or "current_shortcut" not in state:
        print("⚠️ Not executing shortcut, skipping template generation")
        return state

    # Call generate_execution_template function
    state_dict = dict(state)
    execution_template = generate_execution_template(
        state_dict, state["current_shortcut"], state["task"]
    )
    state["execution_template"] = execution_template

    print(
        f"✓ Generated execution template with {len(state['execution_template']['steps'])} steps"
    )

    return state


def execute_action_node(state: DeploymentState) -> DeploymentState:
    """
    Execute operation
    """
    print("\n" + "="*80)
    print("🚀 EXECUTE ACTION NODE - 开始执行操作")
    print(f"  - should_execute_shortcut: {state.get('should_execute_shortcut')}")
    print(f"  - has execution_template: {'execution_template' in state and state['execution_template']}")
    print(f"  - has matched_elements: {bool(state.get('matched_elements'))}")
    if state.get('matched_elements'):
        print(f"  - Matched elements count: {len(state['matched_elements'])}")
        for i, match in enumerate(state['matched_elements'], 1):
            print(f"    {i}. ID: {match.get('screen_element_id')}, "
                  f"Action: {match.get('action_type')}, "
                  f"Score: {match.get('match_score', 0):.4f}")
    print("="*80 + "\n")
    
    state_dict = dict(state)

    if state.get("should_execute_shortcut") and state.get("execution_template"):
        print("🚀 Executing high-level operation...")
        print(f"  - Shortcut: {state.get('current_shortcut', {}).get('name', 'Unknown')}")
        print(f"  - Template steps: {len(state['execution_template'].get('steps', []))}")
        
        # Call execute_high_level_action function
        result = execute_high_level_action(
            state_dict, state["associated_shortcuts"], state["execution_template"]
        )

        if result["status"] == "success":
            print("✨ High-level operation executed successfully!")
            state["execution_status"] = "success"
            state["completed"] = True

            # Update history
            if "execution_history" in result:
                state["history"] = result["execution_history"]

            # Update final screenshot
            if "final_screenshot" in result and "current_page" in state:
                state["current_page"]["screenshot"] = result["final_screenshot"]
        else:
            print(
                f"❌ High-level operation execution failed: {result.get('message', '')}"
            )
            # Mark for fallback on failure
            state["should_fallback"] = True
    
    # Handle case where we have matched elements but no shortcut execution
    elif state.get("matched_elements"):
        print("🎯 Executing matched elements...")
        success = True
        
        for element_match in state["matched_elements"]:
            print(f"  - Executing action on element {element_match.get('screen_element_id')} "
                  f"(score: {element_match.get('match_score', 0):.4f})")
            
            # Execute the action on the matched element
            element_success = execute_element_action(state_dict, element_match)
            
            if not element_success:
                print(f"❌ Failed to execute action on element {element_match.get('screen_element_id')}")
                success = False
                state["should_fallback"] = True
                break
        
        if success:
            print("✅ Successfully executed all matched elements")
            state["execution_status"] = "success"
            state["completed"] = True
    
    # Fall back to task matching if no matched elements
    else:
        print("📝 Attempting to match task with high-level actions...")
        # Call match_task_to_action function
        is_matched, matched_action = match_task_to_action(state_dict, state["task"])

        if is_matched and matched_action:
            print(
                f"✓ Task matched to high-level action: {matched_action.get('name', 'Unknown')}"
            )

            # Save current executing high-level action
            state["current_action"] = matched_action

            # Get action sequence
            element_sequence = matched_action.get("element_sequence", [])
            if isinstance(element_sequence, str):
                try:
                    element_sequence = json.loads(element_sequence)
                except:
                    print(f"❌ Failed to parse element sequence")
                    state["should_fallback"] = True
                    return state

            if not element_sequence or not isinstance(element_sequence, list):
                print(f"❌ Element sequence is empty or incorrectly formatted")
                state["should_fallback"] = True
                return state

            # Update state
            state["current_step"] = 0
            state["total_steps"] = len(element_sequence)
            state["execution_status"] = "running"

            # Execute operation sequence (should actually enter next cycle)
            state["execution_template"] = {"steps": element_sequence}
            # Not executing here, returning to capture_screen for next cycle
        else:
            print(
                "⚠️ Unable to complete task with high-level operations, marking for fallback"
            )
            state["should_fallback"] = True

    return state


def fallback_node(state: DeploymentState) -> DeploymentState:
    """
    Fall back to React mode
    """
    print("⚠️ Falling back to basic operation space")

    # Call fallback_to_react function
    state = fallback_to_react(state)

    # Mark task as completed
    state["completed"] = True

    return state


# Routing functions
def should_fallback(state: DeploymentState) -> str:
    """
    Decide whether to fall back to basic operations
    """
    if state["should_fallback"]:
        return "fallback"
    return "continue"


def should_execute_shortcut(state: DeploymentState) -> str:
    """
    Decide whether to execute shortcut
    """
    if state["should_execute_shortcut"]:
        return "execute_shortcut"
    return "match_task"


def is_task_completed(state: DeploymentState) -> str:
    """
    Check if task is completed
    """
    if state["completed"]:
        return "end"
    return "continue"


# Build state graph
def build_workflow() -> StateGraph:
    """
    Build workflow state graph
    """
    workflow = StateGraph(DeploymentState)

    # Add nodes
    workflow.add_node("capture_screen", capture_screen_node)
    workflow.add_node("match_elements", match_elements_node)
    workflow.add_node("check_shortcuts", check_shortcuts_node)
    workflow.add_node("evaluate_shortcut", shortcut_evaluation_node)
    workflow.add_node("generate_template", generate_template_node)
    workflow.add_node("execute_action", execute_action_node)
    workflow.add_node("fallback", fallback_node)
    workflow.add_node(
        "check_completion", check_task_completion
    )  # New task completion check node

    # Define edges
    workflow.set_entry_point("capture_screen")

    # Routing after screen capture
    workflow.add_conditional_edges(
        "capture_screen",
        should_fallback,
        {"fallback": "fallback", "continue": "match_elements"},
    )

    # Routing after element matching
    workflow.add_conditional_edges(
        "match_elements",
        should_fallback,
        {"fallback": "fallback", "continue": "check_shortcuts"},
    )

    # Check shortcut associations
    workflow.add_edge("check_shortcuts", "evaluate_shortcut")

    # Routing after shortcut evaluation
    workflow.add_conditional_edges(
        "evaluate_shortcut",
        should_execute_shortcut,
        {"execute_shortcut": "generate_template", "match_task": "execute_action"},
    )

    # Execute action after template generation
    workflow.add_edge("generate_template", "execute_action")

    # Check task completion after action execution
    workflow.add_edge("execute_action", "check_completion")

    # Routing after task completion check
    workflow.add_conditional_edges(
        "check_completion",
        is_task_completed,
        {"end": END, "continue": "capture_screen"},
    )

    # Check task completion after fallback
    workflow.add_edge("fallback", "check_completion")

    return workflow


def check_task_completion(state: DeploymentState) -> DeploymentState:
    """
    Determine if task is completed

    Args:
        state: Execution state

    Returns:
        Updated execution state with task completion status
    """
    # Skip judgment if too few steps
    if state["current_step"] < 2:
        return state

    print("🔍 Evaluating if task is completed...")

    # Get task description
    task = state["task"]

    # Step 1: Generate task completion criteria
    completion_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are an assistant that will help analyze task completion criteria. Please carefully read the following user task:",
            ),
            (
                "human",
                f"The user's task is: {task}\nPlease describe clear, checkable task completion criteria. For example: 'When certain elements or states appear on the page, it indicates the task is complete.'",
            ),
        ]
    )

    completion_chain = completion_prompt | model | StrOutputParser()
    completion_criteria = completion_chain.invoke({})

    # Collect recent screenshots
    recent_screenshots = []
    for step in state["history"][-3:]:
        if "screenshot" in step and step["screenshot"]:
            recent_screenshots.append(step["screenshot"])

    if not recent_screenshots:
        if state["current_page"]["screenshot"]:
            recent_screenshots.append(state["current_page"]["screenshot"])

    if not recent_screenshots:
        print("⚠️ No screenshots available, cannot determine if task is complete")
        return state

    # Build image messages
    image_messages = []
    for idx, img_path in enumerate(recent_screenshots, start=1):
        if os.path.exists(img_path):
            with open(img_path, "rb") as f:
                img_data = base64.b64encode(f.read()).decode("utf-8")
            image_messages.append(
                HumanMessage(
                    content=[
                        {"type": "text", "text": f"Here is data for screenshot {idx}:"},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{img_data}"},
                        },
                    ]
                )
            )

    # Step 2: Determine if task is complete
    judgement_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a page assessment assistant that will determine if a task is complete based on completion criteria and current page screenshots. Please only respond with 'yes' or 'no'.",
            ),
            (
                "human",
                f"The completion criteria is: {completion_criteria}\n"
                f"Based on the following screenshots, determine if the task is complete. Note that if screenshots are identical, it may indicate the task cannot proceed, so respond with 'yes' to end the program.",
            ),
        ]
    )

    # Combine all messages
    all_messages = list(judgement_prompt.messages) + image_messages

    # Call LLM for judgment
    judgement_response = model.invoke(all_messages)
    judgement_answer = judgement_response.content.strip()

    # Update task completion status
    if "yes" in judgement_answer.lower() or "complete" in judgement_answer.lower():
        state["completed"] = True
        state["execution_status"] = "completed"
        print(f"✓ Task completed: {judgement_answer}")
    else:
        state["completed"] = False
        print(f"⚠️ Task not completed: {judgement_answer}")

    # Add to history
    state["history"].append(
        {
            "step": state["current_step"],
            "action": "task_completion_check",
            "completion_criteria": completion_criteria,
            "judgement": judgement_answer,
            "status": "success",
            "completed": state["completed"],
        }
    )

    return state
