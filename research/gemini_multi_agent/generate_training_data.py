import os
import json
import time
import random
import re
from typing import List, Dict, Optional
from multiprocessing import Pool, Manager, cpu_count
from api_clients import GeminiUnderstandingClient, GeminiImageGenerator
_shared_lock = None

class SessionManager:

    def __init__(self, session_id: str, data_dir: str, total_turns: int=20):
        self.session_id = session_id
        self.data_dir = data_dir
        self.total_turns = total_turns
        self.memory_bank = []
        self.conversation_log = []
        self.topic_image_ids: Dict[str, List[int]] = {}
        self.last_topic: Optional[str] = None
        self.image_rank_counter = 0
        self.image_id_to_rank: Dict[int, int] = {}
        self.understand_client = GeminiUnderstandingClient()
        self.generate_client = GeminiImageGenerator()
        self.data_vlm_a = []
        self.data_vlm_b = []
        self.topic_pool = self._get_topic_pool()
        self.current_topic = random.choice(self.topic_pool)['name'] if self.topic_pool else None
        self.turn_plan = self._build_turn_plan() if self.total_turns >= 20 else None

    def _get_topic_pool(self) -> List[Dict]:
        """Benchmark 对齐话题池（与 scenario_generator 一致）"""
        return [{'name': 'retail_storefront', 'generate': '现代化零售门店外观，有清晰的招牌和玻璃橱窗', 'edit': '在橱窗里添加两件主推商品和价格标签', 'cross_ref_edit': '回到那家门店，把招牌改成暖色灯箱', 'understand_single': '门店外观的主要视觉元素有哪些？'}, {'name': 'product_studio', 'generate': '电商产品拍摄棚，白色背景、柔光箱和产品台', 'edit': '在产品台上加一个小型电器产品', 'cross_ref_edit': '回到产品棚，把背景改成浅灰色渐变', 'understand_single': '产品棚里有哪些布光设备？'}, {'name': 'office_meeting', 'generate': '商务会议室，长桌、投影幕布和白板', 'edit': '在会议室里加一台开启的投影仪', 'cross_ref_edit': '回到会议室，把白板内容改成流程图', 'understand_single': '会议室里有哪些办公设备？'}, {'name': 'logistics_warehouse', 'generate': '物流仓库，有货架、包装箱和叉车通道', 'edit': '在货架上加几个带条码的纸箱', 'cross_ref_edit': '回到仓库，把地面通道标线改成黄色', 'understand_single': '仓库里主要有哪些存储设施？'}, {'name': 'public_transit', 'generate': '城市公共交通枢纽，有电子时刻表和候车区', 'edit': '在候车区加几排座椅和指示牌', 'cross_ref_edit': '回到交通枢纽，把时刻表显示改成蓝色界面', 'understand_single': '候车区有哪些设施？'}, {'name': 'restaurant_kitchen', 'generate': '餐厅后厨，有不锈钢操作台和烹饪设备', 'edit': '在操作台上加一排备菜盒', 'cross_ref_edit': '回到后厨，把墙面改成浅色瓷砖', 'understand_single': '后厨里有哪些主要设备？'}, {'name': 'cafe_interior', 'generate': '城市咖啡馆内景，有吧台、菜单牌和座位区', 'edit': '在吧台上加一台咖啡机和几只咖啡杯', 'cross_ref_edit': '回到咖啡馆，把菜单牌改成手写风格', 'understand_single': '咖啡馆里有哪些功能分区？'}, {'name': 'classroom', 'generate': '现代教室，有投影幕布、讲台和整齐座位', 'edit': '在讲台上加一台笔记本电脑', 'cross_ref_edit': '回到教室，把黑板内容改成课程大纲', 'understand_single': '教室里有哪些教学设备？'}, {'name': 'library_reading_area', 'generate': '图书馆阅览区，有书架、阅读桌和台灯', 'edit': '在阅读桌上加几本打开的书', 'cross_ref_edit': '回到阅览区，把台灯换成暖光色', 'understand_single': '阅览区里有哪些阅读设施？'}, {'name': 'hospital_lobby', 'generate': '医疗机构大厅，有挂号台、指示牌和等候区', 'edit': '在等候区加几排座椅和叫号屏', 'cross_ref_edit': '回到大厅，把指示牌改成双语样式', 'understand_single': '大厅里有哪些服务设施？'}, {'name': 'bank_service_hall', 'generate': '银行营业大厅，有服务柜台和排队区域', 'edit': '在大厅里加一台自助服务机', 'cross_ref_edit': '回到银行大厅，把排队指示改成电子屏', 'understand_single': '营业大厅有哪些服务区域？'}, {'name': 'smart_home', 'generate': '智能家居客厅，有电视、音箱和智能中控屏', 'edit': '在客厅加一盏智能落地灯', 'cross_ref_edit': '回到智能客厅，把中控屏界面改成深色模式', 'understand_single': '客厅里有哪些智能设备？'}, {'name': 'fitness_center', 'generate': '健身中心，有跑步机、哑铃区和镜墙', 'edit': '在哑铃区加一排整齐摆放的哑铃', 'cross_ref_edit': '回到健身中心，把地面改成深灰色地垫', 'understand_single': '健身中心有哪些训练区域？'}, {'name': 'manufacturing_line', 'generate': '制造工厂生产线，有传送带、机械臂和安全护栏', 'edit': '在传送带上加几个半成品', 'cross_ref_edit': '回到生产线，把安全标识改成黄色', 'understand_single': '生产线包含哪些关键设备？'}, {'name': 'data_center', 'generate': '数据中心机房，有服务器机柜、线缆和指示灯', 'edit': '在机柜上加几条清晰的标签', 'cross_ref_edit': '回到机房，把指示灯改成蓝色', 'understand_single': '机房里有哪些基础设施？'}, {'name': 'transportation_hub_outdoor', 'generate': '城市交通枢纽外观，有公交站牌、共享单车区和人行道', 'edit': '在共享单车区加几辆不同颜色的单车', 'cross_ref_edit': '回到枢纽外观，把站牌改成电子屏', 'understand_single': '户外交通枢纽有哪些设施？'}, {'name': 'tourist_square', 'generate': '城市广场，有喷泉、步道和指示牌', 'edit': '在广场加一些绿植花坛', 'cross_ref_edit': '回到城市广场，把喷泉灯光改成彩色', 'understand_single': '广场里有哪些公共设施？'}, {'name': 'museum_gallery', 'generate': '博物馆展厅，有展柜、展板和导览标识', 'edit': '在展柜旁加一块说明牌', 'cross_ref_edit': '回到展厅，把导览标识改成图标样式', 'understand_single': '展厅里有哪些展示元素？'}, {'name': 'airport_checkin', 'generate': '机场值机区，有服务柜台、航班信息屏和行李推车', 'edit': '在值机区加几条排队栏杆', 'cross_ref_edit': '回到值机区，把信息屏改成深色背景', 'understand_single': '值机区里有哪些服务设施？'}, {'name': 'hotel_lobby', 'generate': '酒店大堂，有前台、沙发休息区和装饰灯', 'edit': '在前台后方加一面装饰墙', 'cross_ref_edit': '回到酒店大堂，把灯光改成暖黄色', 'understand_single': '酒店大堂有哪些功能区域？'}, {'name': 'marketplace_stall', 'generate': '开放式集市摊位，有货架、价签和遮阳棚', 'edit': '在摊位上加几篮新鲜水果', 'cross_ref_edit': '回到集市摊位，把价签改成红色', 'understand_single': '摊位上展示了哪些物品？'}, {'name': 'construction_site', 'generate': '城市施工现场，有脚手架、围挡和安全标识', 'edit': '在现场加一台小型起重设备', 'cross_ref_edit': '回到施工现场，把围挡改成蓝色', 'understand_single': '施工现场有哪些安全设施？'}, {'name': 'agriculture_greenhouse', 'generate': '现代农业温室，有整齐的种植架和滴灌管线', 'edit': '在种植架上加几排绿植', 'cross_ref_edit': '回到温室，把灯光改成白色补光灯', 'understand_single': '温室里有哪些农业设施？'}, {'name': 'cross_border_market', 'generate': '多文化集市街区，有多语种招牌和不同风格摊位', 'edit': '在街区加几面指示旗帜', 'cross_ref_edit': '回到集市街区，把招牌改成统一色系', 'understand_single': '街区里有哪些多文化元素？'}, {'name': 'co_working_space', 'generate': '共享办公空间，有开放工位、会议角和绿植', 'edit': '在开放工位加几台显示器', 'cross_ref_edit': '回到共享办公空间，把地面改成木纹', 'understand_single': '办公空间有哪些功能分区？'}, {'name': 'supermarket_aisle', 'generate': '大型超市货架通道，整齐的商品陈列和价签', 'edit': '在货架上加几种不同包装的饮料', 'cross_ref_edit': '回到超市通道，把价签改成电子标签', 'understand_single': '通道里有哪些商品类别？'}, {'name': 'pharmacy_counter', 'generate': '药店柜台，有药品陈列柜和咨询窗口', 'edit': '在柜台上加一台自助取药机', 'cross_ref_edit': '回到药店柜台，把指引牌改成蓝白色', 'understand_single': '药店里有哪些服务设施？'}, {'name': 'hotel_room', 'generate': '商务酒店客房，有床、书桌和落地窗', 'edit': '在书桌上加一盏阅读台灯', 'cross_ref_edit': '回到酒店客房，把床品改成深灰色', 'understand_single': '客房里有哪些家具？'}, {'name': 'airport_security', 'generate': '机场安检区，有安检通道、传送带和安检门', 'edit': '在安检区加一块排队引导牌', 'cross_ref_edit': '回到安检区，把通道标识改成黄色', 'understand_single': '安检区有哪些设备？'}, {'name': 'train_carriage', 'generate': '高铁车厢内景，有座椅排布和行李架', 'edit': '在车厢里加一块电子到站信息屏', 'cross_ref_edit': '回到车厢，把座椅颜色改成蓝色', 'understand_single': '车厢里有哪些设施？'}, {'name': 'campus_canteen', 'generate': '校园食堂，有取餐窗口、餐桌和标识牌', 'edit': '在餐桌上加几套餐盘', 'cross_ref_edit': '回到食堂，把标识牌改成中英双语', 'understand_single': '食堂里有哪些服务区域？'}, {'name': 'car_showroom', 'generate': '汽车展厅，有展示车辆、玻璃幕墙和接待区', 'edit': '在展厅里加一块车型介绍屏', 'cross_ref_edit': '回到汽车展厅，把地面改成深色地砖', 'understand_single': '展厅里有哪些展示元素？'}, {'name': 'bank_atm_area', 'generate': '银行自助区，有多台ATM机和安全隔挡', 'edit': '在自助区加一块使用说明牌', 'cross_ref_edit': '回到自助区，把隔挡改成透明玻璃', 'understand_single': '自助区有哪些设备？'}, {'name': 'community_service_center', 'generate': '社区服务中心大厅，有咨询台、等候区和公告栏', 'edit': '在公告栏上加几张活动海报', 'cross_ref_edit': '回到服务中心，把等候区座椅改成蓝色', 'understand_single': '服务中心有哪些功能区？'}, {'name': 'museum_ticketing', 'generate': '博物馆售票处，有售票窗口和电子票机', 'edit': '在售票处加一块票价信息屏', 'cross_ref_edit': '回到售票处，把排队栏杆改成金属色', 'understand_single': '售票处有哪些服务设施？'}, {'name': 'renewable_energy_farm', 'generate': '可再生能源场景，有大片太阳能板和远处风力发电机', 'edit': '在太阳能板旁加一条检修道路', 'cross_ref_edit': '回到能源场景，把天空改成晴朗蓝天', 'understand_single': '场景里有哪些能源设施？'}, {'name': 'container_port', 'generate': '集装箱港口，有堆叠的集装箱和龙门吊', 'edit': '在码头加一艘停靠的货轮', 'cross_ref_edit': '回到港口，把集装箱颜色改成红蓝混搭', 'understand_single': '港口里有哪些装卸设备？'}, {'name': 'highway_interchange', 'generate': '城市高速立交，有多层车道和交通指示牌', 'edit': '在道路旁加几块限速标志', 'cross_ref_edit': '回到立交，把路面标线改成白色', 'understand_single': '立交里有哪些交通元素？'}, {'name': 'urban_park', 'generate': '城市公园，有草坪、步道和休息长椅', 'edit': '在公园里加一片花坛和喷泉', 'cross_ref_edit': '回到公园，把喷泉灯光改成暖色', 'understand_single': '公园里有哪些公共设施？'}, {'name': 'mountain_trail', 'generate': '山地徒步小径，有指示牌、木栈道和远处山峰', 'edit': '在小径旁加几块里程标识', 'cross_ref_edit': '回到徒步小径，把天空改成清晨薄雾', 'understand_single': '小径周围有哪些自然元素？'}, {'name': 'river_bridge', 'generate': '跨河大桥，桥面车道和河面倒影清晰可见', 'edit': '在桥上加一排路灯', 'cross_ref_edit': '回到大桥，把路灯改成暖黄色', 'understand_single': '大桥有哪些结构元素？'}, {'name': 'sports_arena', 'generate': '体育场馆内景，有看台、球场和大屏幕', 'edit': '在场馆中加几排旗帜装饰', 'cross_ref_edit': '回到体育场，把场地颜色改成更鲜明的绿色', 'understand_single': '场馆里有哪些主要设施？'}, {'name': 'astronomy_observatory', 'generate': '天文观测台，有圆顶建筑和大型望远镜', 'edit': '在观测台旁加一块信息展示牌', 'cross_ref_edit': '回到观测台，把天空改成繁星夜空', 'understand_single': '观测台里有哪些设备？'}, {'name': 'electronics_assembly', 'generate': '电子装配线，有工作台、线路板和防静电设施', 'edit': '在工作台上加几块半成品电路板', 'cross_ref_edit': '回到装配线，把台面改成蓝色防静电垫', 'understand_single': '装配线有哪些关键设备？'}, {'name': 'quality_inspection_lab', 'generate': '质量检测实验室，有检测台、放大镜和测量工具', 'edit': '在检测台上加一个正在检测的零部件', 'cross_ref_edit': '回到实验室，把照明改成冷白色', 'understand_single': '实验室里有哪些检测工具？'}, {'name': 'warehouse_robotics', 'generate': '智能仓储区，有自动搬运车和货架', 'edit': '在通道里加几条地面导航线', 'cross_ref_edit': '回到仓储区，把导航线改成蓝色', 'understand_single': '仓储区有哪些自动化设施？'}, {'name': 'aerial_farmland', 'generate': '农田航拍视角，有规则的耕作格和灌溉渠道', 'edit': '在农田里加一条农机作业轨迹', 'cross_ref_edit': '回到农田，把作物颜色改成深绿色', 'understand_single': '农田里有哪些耕作元素？'}, {'name': 'coastal_breakwater', 'generate': '海岸防波堤，有礁石、堤坝和海浪', 'edit': '在堤坝旁加一个灯塔', 'cross_ref_edit': '回到海岸，把海面改成浅蓝色', 'understand_single': '海岸场景有哪些防护设施？'}, {'name': 'city_residential_street', 'generate': '城市居民街道，有行道树、路灯和住宅楼', 'edit': '在街道上加几辆停靠的汽车', 'cross_ref_edit': '回到居民街道，把路灯改成暖黄色', 'understand_single': '街道上有哪些公共设施？'}, {'name': 'cultural_center', 'generate': '文化中心外观，有开放广场和引导标识', 'edit': '在广场加几组公共座椅', 'cross_ref_edit': '回到文化中心，把标识牌改成双语样式', 'understand_single': '文化中心周边有哪些公共设施？'}, {'name': 'forest_wildlife', 'generate': '森林生态场景，有高大树木、溪流和野生动物足迹', 'edit': '在林间加几只觅食的鹿', 'cross_ref_edit': '回到森林场景，把溪流改成更清澈', 'understand_single': '森林里有哪些自然元素？'}, {'name': 'savanna_wildlife', 'generate': '稀树草原，有金色草地和零散的乔木', 'edit': '在草原上加一群斑马', 'cross_ref_edit': '回到草原，把天空改成傍晚色调', 'understand_single': '草原里有哪些生物或地貌？'}, {'name': 'coral_reef', 'generate': '珊瑚礁生态，有多样珊瑚和清澈海水', 'edit': '在珊瑚礁附近加几群小型热带鱼', 'cross_ref_edit': '回到珊瑚礁，把水体光线改成更明亮', 'understand_single': '珊瑚礁里有哪些生物？'}, {'name': 'rainforest_canopy', 'generate': '热带雨林树冠层，有茂密叶片和薄雾', 'edit': '在树冠层加几只彩色小鸟', 'cross_ref_edit': '回到雨林树冠，把雾气改成更轻薄', 'understand_single': '雨林场景包含哪些生态要素？'}, {'name': 'microscope_cells', 'generate': '显微镜视角下的细胞结构，有清晰的细胞膜和细胞核', 'edit': '在画面中加一些染色的颗粒结构', 'cross_ref_edit': '回到细胞图，把背景改成浅灰色', 'understand_single': '细胞图里能看到哪些结构？'}, {'name': 'plant_growth_lab', 'generate': '植物生长实验环境，有培养架、补光灯和营养液', 'edit': '在培养架上加一排幼苗', 'cross_ref_edit': '回到实验环境，把灯光改成蓝紫色', 'understand_single': '实验环境里有哪些设备？'}, {'name': 'lunar_surface', 'generate': '月球表面，有灰色地表、陨石坑和远处地球', 'edit': '在地表加一台月球探测车', 'cross_ref_edit': '回到月球表面，把地球位置改到地平线附近', 'understand_single': '月球场景有哪些地貌特征？'}, {'name': 'mars_outpost', 'generate': '火星基地外景，有红色地表、模块化基地和天线', 'edit': '在基地旁加一台无人探测车', 'cross_ref_edit': '回到火星基地，把天线改成阵列式', 'understand_single': '火星场景里有哪些设施？'}, {'name': 'satellite_orbit', 'generate': '地球轨道上的卫星，有太阳能板和地球背景', 'edit': '在卫星旁加一个小型对接舱', 'cross_ref_edit': '回到卫星画面，把太阳能板改成更大尺寸', 'understand_single': '轨道场景里有哪些航天元素？'}, {'name': 'galaxy_nebula', 'generate': '星云和银河场景，有彩色气体云和星点', 'edit': '在星云附近加一片更亮的星团', 'cross_ref_edit': '回到星云场景，把主色调改成蓝紫色', 'understand_single': '星云场景里有哪些天体特征？'}]

    def _build_turn_plan(self) -> List[Dict]:
        """生成 20 轮的难检索策略计划（跨话题/长期记忆/负样本/多图对比）"""
        if not self.topic_pool:
            return []
        rng = random.Random()
        plan: List[Dict] = []
        topics = rng.sample(self.topic_pool, min(5, len(self.topic_pool)))
        main_topics = topics[:4]
        extra_topic = topics[4] if len(topics) >= 5 else rng.choice(main_topics)
        for idx, topic in enumerate(main_topics):
            plan.append({'action': 'generate', 'topic': topic, 'negative': False})
            plan.append({'action': 'edit', 'topic': topic, 'cross_topic': False})
            if rng.random() < 0.25:
                plan.append({'action': 'understand', 'topic': topic, 'mode': 'single'})
        if rng.random() < 0.7 and len(plan) + 2 <= self.total_turns:
            plan.append({'action': 'generate', 'topic': extra_topic, 'negative': False})
            plan.append({'action': 'edit', 'topic': extra_topic, 'cross_topic': False})
        negative_added = False
        compare_count = 0
        while len(plan) < self.total_turns:
            remaining = self.total_turns - len(plan)
            if remaining <= 2:
                if compare_count < 2:
                    plan.append({'action': 'understand', 'mode': 'compare', 'cross_topic': True, 'long_term': True})
                    compare_count += 1
                    continue
                if not negative_added:
                    plan.append({'action': 'generate', 'negative': True})
                    negative_added = True
                    continue
            r = rng.random()
            if r < 0.25:
                plan.append({'action': 'edit', 'mode': 'cross_edit', 'cross_topic': True, 'long_term': rng.random() < 0.5})
            elif r < 0.5:
                plan.append({'action': 'understand', 'mode': 'single', 'cross_topic': True, 'long_term': rng.random() < 0.5})
            elif r < 0.7:
                plan.append({'action': 'understand', 'mode': 'compare', 'cross_topic': True, 'long_term': True})
                compare_count += 1
            elif r < 0.85:
                plan.append({'action': 'generate', 'negative': True})
                negative_added = True
            else:
                plan.append({'action': 'edit', 'mode': 'cross_edit', 'cross_topic': True, 'long_term': True})
        return plan[:self.total_turns]

    def _register_topic_image(self, topic_name: Optional[str], mem_id: int):
        if not topic_name:
            return
        if topic_name not in self.topic_image_ids:
            self.topic_image_ids[topic_name] = []
        self.topic_image_ids[topic_name].append(mem_id)

    def _register_image_rank(self, mem_id: int) -> int:
        if mem_id in self.image_id_to_rank:
            return self.image_id_to_rank[mem_id]
        rank = self.image_rank_counter
        self.image_id_to_rank[mem_id] = rank
        self.image_rank_counter += 1
        return rank

    def _describe_image_for_refine(self, img_path: str) -> str:
        """轻量描述，用于改写用户指令（不写入VLM-A）"""
        prompt = '请用中文一句话简要描述这张图片的主要内容。'
        try:
            desc = self.understand_client.chat(prompt, images=[img_path], temperature=0.2)
            return desc.strip() if desc else ''
        except Exception:
            return ''

    def save_image(self, img_bytes: bytes, turn_id: int, suffix: str='') -> str:
        """保存图片并返回路径"""
        img_name = f"sess_{self.session_id}_turn_{turn_id}{('_' + suffix if suffix else '')}.jpg"
        img_path = os.path.join(self.data_dir, 'images', img_name)
        os.makedirs(os.path.dirname(img_path), exist_ok=True)
        with open(img_path, 'wb') as f:
            f.write(img_bytes)
        return img_path

    def generate_user_uploaded_image(self, turn_id: int) -> Optional[str]:
        """
        模拟用户上传图片：先用Gemini生成一张图，作为“用户上传”的图
        """
        topic_hint = random.choice(self.topic_pool)['generate'] if self.topic_pool else 'a random daily life photo'
        prompt = f'A random daily life photo or a specific object related to {topic_hint}, realistic style.'
        try:
            img_bytes = self.generate_client.generate(prompt=prompt, images=None, image_format='base64', max_retries=10, retry_delay=3.0)
            if not img_bytes:
                return None
            img_path = self.save_image(img_bytes, turn_id, suffix='user_upload')
            print(f'  用户上传图片已保存: {img_path}')
            return img_path
        except Exception as e:
            print(f'  错误: 生成用户上传图片时发生异常: {e}')
            return None

    def create_memory(self, img_path: str, turn_id: int, is_user_uploaded: bool=False, update_info: str=None, is_edit: bool=False, ref_image_paths: Optional[List[str]]=None, edit_instruction: Optional[str]=None, topic_name: Optional[str]=None) -> Dict:
        """
        为图片生成或修正Memory Item
        """
        if update_info:
            prompt = f"用户提供了补充/修正信息：'{update_info}'。请结合图片内容，重新生成准确的结构化信息。"
        else:
            prompt = '请分析这张图片，为Memory Bank生成结构化信息。'
        if is_edit:
            diff_rule = '请生成 diff 字段，描述这张新图相对于参考图的明显特点。注意：diff 中只描述新图的特点，不要描述参考图的特点，也不要提到参考图。'
            if edit_instruction:
                diff_rule += f'\n编辑指令：{edit_instruction}'
        else:
            diff_rule = 'diff 字段留空字符串即可。'
        prompt += f'\n要求：description 尽量详细。\n语言要求：所有字段内容必须是中文（title/description/diff），不要输出英文或中英混杂。\n{diff_rule}\n\n输出必须是有效的JSON格式：\n{{\n    "title": "简短的标题",\n    "description": "详细的画面描述",\n    "diff": "仅用于edit任务的差异描述，非edit时为空字符串"\n}}\n只输出JSON，不要有其他内容。\n'
        prompt_images = [img_path]
        if is_edit and ref_image_paths:
            prompt_images = ref_image_paths + [img_path]
        response_text = self.understand_client.chat(prompt, images=prompt_images, temperature=0.1)
        try:
            clean_json = re.sub('```json\\s*|```', '', response_text).strip()
            mem_data = json.loads(clean_json)
        except:
            mem_data = {'title': 'Error', 'description': response_text[:200], 'diff': ''}
        memory_item = {'id': turn_id, 'path': img_path, 'memory': mem_data, 'topic': topic_name}
        self.data_vlm_a.append({'session_id': self.session_id, 'turn_id': turn_id // 1000 if turn_id >= 1000 else turn_id, 'image_path': img_path, 'ground_truth_memory': mem_data, 'is_user_uploaded': is_user_uploaded})
        return memory_item

    def user_simulator(self, turn_id: int) -> Dict:
        """
        模拟用户行为，决定本轮动作、指代风格和具体指令
        """
        plan = self.turn_plan[turn_id] if self.turn_plan and turn_id < len(self.turn_plan) else None
        actions = ['generate', 'edit', 'understand']
        if turn_id == 0:
            weights = [0.45, 0.3, 0.25]
        else:
            weights = [0.35, 0.4, 0.25]
        chosen_action = plan.get('action') if plan else random.choices(actions, weights=weights)[0]
        referral_style = 'index' if random.random() < 0.2 else 'feature'
        memory_descriptions = [f"[{i + 1}] {m['memory'].get('title', 'Unknown')} - {m['memory'].get('description', '')}" for i, m in enumerate(self.memory_bank)]
        memory_summary = '\n'.join(memory_descriptions) if memory_descriptions else '（当前对话刚开始，没有任何历史图片）'
        action_prompts = {'generate': '任务：【绘画请求】（文生图）。\n要求：只通过文字描述生成新图，不引用历史图片（ref_image_indices 为 []）。', 'edit': '任务：【对图片的编辑/修改】。\n要求：输入 1-3 张图片进行编辑，数量随机。可以要求增删物体、改变颜色、更换背景等。', 'understand': '任务：【针对图片的提问/评价】（MMU任务）。\n要求：输入 0-3 张图片进行提问，数量随机。'}
        style_req = "【强制指代要求】：必须使用数字索引指代图片，如'第1张'、'最后一张'、'那张[2]'。" if referral_style == 'index' else "【强制指代要求】：严禁使用数字索引，必须使用视觉特征（'那个红色的'）或模糊代词（'刚生成的那个'）。"
        plan_topic = plan.get('topic') if plan else None
        plan_topic_name = plan_topic.get('name') if isinstance(plan_topic, dict) else None
        topic_hint = ''
        if plan_topic and isinstance(plan_topic, dict):
            if chosen_action == 'generate':
                topic_hint = plan_topic.get('generate', '')
            elif chosen_action == 'edit':
                if plan.get('cross_topic'):
                    topic_hint = plan_topic.get('cross_ref_edit', plan_topic.get('edit', ''))
                else:
                    topic_hint = plan_topic.get('edit', '')
            elif chosen_action == 'understand':
                topic_hint = plan_topic.get('understand_single', '')
        extra_constraints = []
        if plan and plan.get('negative'):
            extra_constraints.append('【本轮为负样本】：必须是全新场景，不能引用任何历史图片。')
        if plan and plan.get('mode') == 'compare':
            extra_constraints.append('【多图对比】：需要同时引用两张不同话题的图片进行对比提问。')
        if plan and plan.get('cross_topic'):
            extra_constraints.append('【跨话题检索】：倾向回到较早或不同话题的图片。')
        if plan and plan.get('long_term'):
            extra_constraints.append('【长期记忆】：尽量回到较早生成的图片。')
        hint_text = '\n'.join([h for h in [topic_hint] if h])
        constraint_text = '\n'.join(extra_constraints) if extra_constraints else ''
        user_prompt = f'\n你是一个正在与AI绘图助手对话的用户。你说话很随性，不会像机器人一样死板。\n历史记录：\n{memory_summary}\n\n当前任务目标：\n{action_prompts[chosen_action]}\n{hint_text}\n{constraint_text}\n\n{style_req}\n\n请输出一个JSON，其中的 "instruction" 字段必须是你（用户）直接说出口的话。\n【特殊规则】：\n- 如果任务是 edit 或 understand 且历史记录为空，你必须假设你正在【上传一张新图】并围绕它说话。\n- 绝对禁止包含任何分析、解析、思路、或“我的指令是”之类的废话。直接输出你想说的话。\n\n输出JSON格式：\n{{\n    "action_type": "{chosen_action}",\n    "edit_subtype": "如果是edit任务请填写子类型(style_change/add_object/remove_object/attr_change/bg_change)，否则为null",\n    "ref_image_indices": [你引用的历史图片索引列表（1开始），如果没有引用历史图则为[]],\n    "instruction": "你直接说的话",\n    "should_upload_image": true/false (如果是第一轮且任务需要图片，必须为true；其他情况30%概率为true)\n}}\n'
        response = self.understand_client.chat(user_prompt, images=[self.memory_bank[-1]['path']] if self.memory_bank else [])

        def _ids_for_topic(topic_name: str) -> List[int]:
            return [m['id'] for m in self.memory_bank if m.get('topic') == topic_name]

        def _first_id_for_topic(topic_name: str) -> Optional[int]:
            ids = _ids_for_topic(topic_name)
            return ids[0] if ids else None

        def _pick_topic_with_history(exclude: Optional[str]=None) -> Optional[str]:
            topics = list(self.topic_image_ids.keys())
            if not topics:
                return None
            if exclude and len(topics) > 1:
                candidates = [t for t in topics if t != exclude]
                if candidates:
                    return random.choice(candidates)
            return random.choice(topics)
        try:
            clean_json = re.sub('```json\\s*|```', '', response).strip()
            data = json.loads(clean_json)
            data['action_type'] = chosen_action
            instruction_text = data.get('instruction', '')
            desired_k = None
            if chosen_action == 'generate':
                desired_k = 0
                data['ref_image_indices'] = []
                data['should_upload_image'] = False
            elif chosen_action == 'edit':
                desired_k = random.choices([1, 2, 3], weights=[4, 2, 1])[0]
            elif chosen_action == 'understand':
                desired_k = random.choices([0, 1, 2, 3], weights=[2, 4, 1, 1])[0]
            if plan and plan.get('negative'):
                desired_k = 0
            if plan and plan.get('mode') == 'compare':
                desired_k = 2
            else:
                compare_keywords = ['对比', '比较', '差异', '不同', '区别']
                image_keywords = ['这张', '那张', '刚才', '上一张', '上张', '第', '图', '照片', '画面', '这幅']
                single_keywords = ['只用', '仅用', '就用', '只需要', '就这张', '这一张', '这张图', '上一张', '刚才那张', '那张']
                two_keywords = ['两张', '二张', '俩图', '双图', '2张', '2幅', '2个', '两个', '两幅']
                three_keywords = ['三张', '三幅', '三个', '3张', '3幅', '3个', '三张图', '三张图片']
                blend_keywords = ['合成', '融合', '结合', '拼接', '拼在', '合并', '混合', '叠加']
                bg_swap_keywords = ['背景换成', '背景替换', '换成背景', '换到', '放到', '放在', '置于']
                if any((k in instruction_text for k in three_keywords)):
                    desired_k = 3
                elif any((k in instruction_text for k in two_keywords)):
                    desired_k = 2
                elif any((k in instruction_text for k in single_keywords)):
                    desired_k = 1
                elif any((k in instruction_text for k in compare_keywords)):
                    desired_k = 2
                elif any((k in instruction_text for k in blend_keywords + bg_swap_keywords)):
                    desired_k = max(desired_k or 0, 2)
                elif any((k in instruction_text for k in image_keywords)):
                    desired_k = 1
            if turn_id == 0 and chosen_action in ['edit', 'understand']:
                data['should_upload_image'] = True
            gt_ids: List[int] = []
            if desired_k and desired_k > 0:
                if plan and plan.get('mode') == 'compare':
                    topics = list(self.topic_image_ids.keys())
                    if len(topics) >= 2:
                        t1, t2 = random.sample(topics, 2)
                        for t in [t1, t2]:
                            first_id = _first_id_for_topic(t)
                            if first_id is not None:
                                gt_ids.append(first_id)
                    elif len(topics) == 1:
                        first_id = _first_id_for_topic(topics[0])
                        if first_id is not None:
                            gt_ids.append(first_id)
                    if len(gt_ids) < desired_k:
                        all_ids = [m['id'] for m in self.memory_bank]
                        random.shuffle(all_ids)
                        for mid in all_ids:
                            if mid not in gt_ids:
                                gt_ids.append(mid)
                            if len(gt_ids) >= desired_k:
                                break
                else:
                    target_topic = plan_topic_name
                    if plan and plan.get('cross_topic') and (not target_topic):
                        target_topic = _pick_topic_with_history(exclude=self.last_topic)
                    if plan and plan.get('long_term') and target_topic:
                        first_id = _first_id_for_topic(target_topic)
                        if first_id is not None:
                            gt_ids = [first_id]
                    if not gt_ids:
                        candidates = _ids_for_topic(target_topic) if target_topic else [m['id'] for m in self.memory_bank]
                        random.shuffle(candidates)
                        gt_ids = candidates[:desired_k]
            user_uploaded_images = []
            need_upload = data.get('should_upload_image', False)
            if chosen_action == 'edit' and desired_k is not None:
                missing = max(0, desired_k - len(gt_ids))
                need_upload = need_upload or missing > 0
            if chosen_action == 'understand' and desired_k is not None:
                missing = max(0, desired_k - len(gt_ids))
                need_upload = need_upload or missing > 0
            while need_upload:
                uploaded = self.generate_user_uploaded_image(turn_id)
                if not uploaded:
                    break
                user_uploaded_images.append(uploaded)
                up_id = (turn_id + 1) * 1000 + len(user_uploaded_images) - 1
                gt_ids.append(up_id)
                if chosen_action in ['edit', 'understand']:
                    upload_desc = self._describe_image_for_refine(uploaded)
                    refine_prompt = f'''\n你现在是一个真实的用户。你刚才想说："{data.get('instruction')}"。\n但现在你又上传了一张参考图，描述为：{upload_desc}。\n请以用户的口吻重说一遍你的要求，要极其自然地提到你刚上传的这张图。不要废话，只输出你说的话。\n禁止出现“发错图/传错/不好意思/抱歉/对不起/道歉/不小心/弄错/误传/手滑”等表达。\n'''
                    new_instr = self.understand_client.chat(refine_prompt, images=[uploaded])
                    if new_instr:
                        data['instruction'] = re.sub('^(修改后的)?指令[:：]|["”“”]|解析.*|思路.*', '', new_instr, flags=re.S).strip()
                if chosen_action == 'edit' and desired_k is not None:
                    if len(gt_ids) >= desired_k:
                        break
                elif chosen_action == 'understand' and desired_k is not None:
                    if len(gt_ids) >= desired_k:
                        break
                else:
                    break
            if gt_ids:
                deduped = []
                seen = set()
                for gid in gt_ids:
                    if gid not in seen:
                        deduped.append(gid)
                        seen.add(gid)
                gt_ids = deduped
                if desired_k is not None and len(gt_ids) > desired_k:
                    gt_ids = gt_ids[:desired_k]
            turn_topic = None
            if chosen_action == 'generate':
                if plan_topic_name:
                    turn_topic = plan_topic_name
                elif self.topic_pool:
                    turn_topic = random.choice(self.topic_pool)['name']
            elif chosen_action in ['edit', 'understand']:
                if plan_topic_name:
                    turn_topic = plan_topic_name
                elif plan and plan.get('cross_topic'):
                    turn_topic = _pick_topic_with_history(exclude=self.last_topic)
                else:
                    turn_topic = self.last_topic
            return {'action_type': chosen_action, 'edit_subtype': data.get('edit_subtype'), 'instruction': data.get('instruction', '继续'), 'ground_truth_ref_image_ids': gt_ids, 'user_uploaded_images': user_uploaded_images, 'topic_for_turn': turn_topic}
        except:
            return {'action_type': chosen_action, 'instruction': '继续', 'ground_truth_ref_image_ids': [], 'user_uploaded_images': [], 'topic_for_turn': None}

    def instruction_translator(self, user_instruction: str, ref_img_paths: List[str], history: List[Dict], edit_type: str=None) -> str:
        """指令翻译器"""
        all_images = [p for p in ref_img_paths if p]
        context = f'任务子类型：{edit_type}。' if edit_type else ''
        history_str = '\n'.join([f"{('用户' if h['role'] == 'user' else '助手')}: {h['content']}" for h in history[-6:]])
        prompt = f'\n对话历史：\n{history_str}\n\n当前用户指令：{user_instruction}\n{context}\n\n请基于提供的参考图片和对话历史，写一段极其详细的英文Image Generation Prompt。\n要求描述画面主体、细节、风格、光影。\n\n【输出格式要求】：\n你必须先进行思考，分析用户指代的具体对象及其在参考图中的特征，然后给出最终的 Prompt。\n格式如下：\n<think>\n（你的思考过程，分析用户指代的历史图片和具体元素）\n</think>\n<ans>\n(最终的英文 Prompt 文本)\n</ans>\n'
        response = self.understand_client.chat(prompt, images=all_images if all_images else None, temperature=0.2)
        return response if response else '<think>None</think><ans>An image based on: ' + user_instruction + '</ans>'

    def _strip_tags(self, text: str) -> str:
        """从 VLM C 输出中提取 <ans> 内容"""
        ans_match = re.search('<ans>(.*?)</ans>', text, re.S)
        if ans_match:
            return ans_match.group(1).strip()
        cleaned = re.sub('<think>.*?</think>', '', text, flags=re.S).strip()
        cleaned = re.sub('<ans>|</ans>', '', cleaned).strip()
        return cleaned

    def run_turn(self, turn_id: int):
        """执行一轮对话"""
        print(f'\n[Session {self.session_id}] Turn {turn_id}')
        instr_data = self.user_simulator(turn_id)
        action = instr_data['action_type']
        instr = instr_data['instruction']
        gt_ids = instr_data['ground_truth_ref_image_ids']
        topic_for_turn = instr_data.get('topic_for_turn')
        uploaded_paths = instr_data.get('user_uploaded_images', [])
        seen_uploads = set()
        for i, upath in enumerate(uploaded_paths):
            if not upath or upath in seen_uploads:
                continue
            seen_uploads.add(upath)
            up_id = (turn_id + 1) * 1000 + i
            up_mem = self.create_memory(upath, up_id, is_user_uploaded=True)
            self.memory_bank.append(up_mem)
            self._register_topic_image(up_mem.get('topic'), up_id)
            self._register_image_rank(up_id)
            if up_id not in gt_ids:
                gt_ids.append(up_id)
        gt_ranks = [self.image_id_to_rank[i] for i in gt_ids if i in self.image_id_to_rank]
        self.data_vlm_b.append({'session_id': self.session_id, 'turn_id': turn_id, 'user_query': instr, 'memory_bank': [{'id': self.image_id_to_rank[m['id']], 'summary': m['memory']} for m in self.memory_bank if m['id'] in self.image_id_to_rank], 'ground_truth_image_ids': gt_ranks})
        if action == 'understand':
            ref_paths = [m['path'] for m in self.memory_bank if m['id'] in gt_ids]
            retrieved_mems = []
            for m_id in gt_ids:
                mem_item = next((m for m in self.memory_bank if m['id'] == m_id), None)
                if mem_item:
                    retrieved_mems.append({'id': m_id, 'memory': mem_item['memory']})
            history_str = '\n'.join([f"{('用户' if h['role'] == 'user' else '助手')}: {h['content']}" for h in self.conversation_log[-6:]])
            mem_texts = [f"记忆[{m['id']}]: {m['memory'].get('title', '')} - {m['memory'].get('description', '')}" for m in retrieved_mems]
            mem_context = '检索到的记忆：\n' + '\n'.join(mem_texts) + '\n\n' if mem_texts else ''
            mmu_prompt = f'\n对话历史：\n{history_str}\n\n{mem_context}当前用户指令：{instr}\n任务类型：mmu\n请根据指令和参考图片进行思考，给出详细、专业且精炼的回答。使用 <think> 标签进行思考，使用 <ans> 标签输出结果。\n'
            raw_ans = self.understand_client.chat(mmu_prompt, images=ref_paths if ref_paths else None)
            clean_ans = self._strip_tags(raw_ans)
            self.conversation_log.append({'turn': turn_id, 'role': 'user', 'content': instr})
            self.conversation_log.append({'turn': turn_id, 'role': 'assistant', 'content': clean_ans})
            return True
        ref_paths = [m['path'] for m in self.memory_bank if m['id'] in gt_ids]
        retrieved_mems = []
        for m_id in gt_ids:
            mem_item = next((m for m in self.memory_bank if m['id'] == m_id), None)
            if mem_item:
                retrieved_mems.append({'id': m_id, 'memory': mem_item['memory']})
        raw_gen_prompt = self.instruction_translator(instr, ref_paths, self.conversation_log, edit_type=instr_data.get('edit_subtype'))
        clean_gen_prompt = self._strip_tags(raw_gen_prompt)
        img_bytes = self.generate_client.generate(prompt=clean_gen_prompt, images=ref_paths if ref_paths else None)
        if not img_bytes:
            return False
        img_path = self.save_image(img_bytes, turn_id)
        if not topic_for_turn and gt_ids:
            first_ref = next((m for m in self.memory_bank if m['id'] == gt_ids[0]), None)
            if first_ref:
                topic_for_turn = first_ref.get('topic')
        new_mem = self.create_memory(img_path, turn_id, is_edit=action == 'edit', ref_image_paths=ref_paths if action == 'edit' else None, edit_instruction=instr if action == 'edit' else None, topic_name=topic_for_turn)
        self.memory_bank.append(new_mem)
        self._register_topic_image(new_mem.get('topic'), new_mem['id'])
        self._register_image_rank(new_mem['id'])
        if new_mem.get('topic'):
            self.last_topic = new_mem.get('topic')
        self.conversation_log.append({'turn': turn_id, 'role': 'user', 'content': instr})
        self.conversation_log.append({'turn': turn_id, 'role': 'assistant', 'content': {'path': img_path, 'prompt': clean_gen_prompt}})
        return True

    def save_data(self):
        """保存数据"""
        global _shared_lock
        files = {f'train_vlm_a_{self.session_id}.jsonl': self.data_vlm_a, f'train_vlm_b_{self.session_id}.jsonl': self.data_vlm_b}
        if _shared_lock:
            with _shared_lock:
                for fname, dataset in files.items():
                    with open(os.path.join(self.data_dir, fname), 'a', encoding='utf-8') as f:
                        for item in dataset:
                            f.write(json.dumps(item, ensure_ascii=False) + '\n')
        with open(os.path.join(self.data_dir, f'log_{self.session_id}.json'), 'w', encoding='utf-8') as f:
            json.dump({'session_id': self.session_id, 'history': self.conversation_log}, f, ensure_ascii=False, indent=2)

def run_single_session(session_id: str, turns_per_session: int, data_dir: str, delay: float, lock=None):
    global _shared_lock
    _shared_lock = lock
    try:
        session = SessionManager(session_id=session_id, data_dir=data_dir, total_turns=turns_per_session)
        for turn in range(turns_per_session):
            if not session.run_turn(turn):
                break
            time.sleep(delay)
        session.save_data()
        return session_id
    except Exception as e:
        print(f'Error session {session_id}: {e}')
        return None

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_sessions', type=int, default=1)
    parser.add_argument('--turns_per_session', type=int, default=5)
    parser.add_argument('--data_dir', type=str, default='./simulated_data')
    parser.add_argument('--delay', type=float, default=2.0)
    parser.add_argument('--num_processes', type=int, default=None)
    args = parser.parse_args()
    start_time = time.time()
    start_time_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))
    print(f'开始时间: {start_time_str}')
    num_processes = args.num_processes or cpu_count()
    manager = Manager()
    lock = manager.Lock()
    pool = Pool(processes=num_processes)
    for i in range(args.num_sessions):
        sess_id = f'{i + 1:04d}'
        pool.apply_async(run_single_session, args=(sess_id, args.turns_per_session, args.data_dir, args.delay, lock))
    pool.close()
    pool.join()
    end_time = time.time()
    end_time_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time))
    duration = end_time - start_time
    print('\n所有Session生成完毕！')
    print(f'结束时间: {end_time_str}')
    print(f'总计耗时: {duration:.2f} 秒 ({duration / 60:.2f} 分钟)')
    if args.num_sessions > 0:
        print(f'平均每个Session耗时: {duration / args.num_sessions:.2f} 秒')
if __name__ == '__main__':
    main()
