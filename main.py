# ╭──────────────────────────────────────────────────────╮
# │                                                      │
# │   ██╗     ██╗  ██╗██╗  ████████╗██╗  ██╗             │
# │   ██║     ╚██╗██╔╝██║  ╚══██╔══╝╚██╗██╔╝             │
# │   ██║      ╚███╔╝ ██║     ██║    ╚███╔╝              │
# │   ██║      ██╔██╗ ██║     ██║    ██╔██╗              │
# │   ███████╗██╔╝ ██╗███████╗██║   ██╔╝ ██╗             │
# │   ╚══════╝╚═╝  ╚═╝╚══════╝╚═╝   ╚═╝  ╚═╝             │
# │                                                      │
# │   Author: LXLTX-Lab                                  │
# │   GitHub: https://github.com/lxltx2025               │
# │   Date: 2025-12-25                                   │
# │   License: MIT                                       │
# │                                                      │
# ╰──────────────────────────────────────────────────────╯

"""
国自然科学基金申请书写作助手 - 主入口
"""

# ===== 环境配置（必须在最开始） =====
import os
import sys
import warnings

# 禁用所有警告
warnings.filterwarnings("ignore")

# 设置环境变量
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["CHROMA_TELEMETRY"] = "False"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"

# Windows特定：禁用分布式警告
if os.name == 'nt':
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"

# ===== 正常导入 =====
import argparse
from pathlib import Path

# 确保src目录在路径中
sys.path.insert(0, str(Path(__file__).parent))


def cmd_process_data(args):
    """处理训练数据"""
    from src.data_processor import DataProcessor
    
    processor = DataProcessor(
        min_content_length=args.min_length,
        quality_threshold=args.quality_threshold
    )
    
    processor.run(
        input_dir=args.input_dir,
        output_path=args.output
    )


def cmd_train(args):
    """微调模型"""
    from src.trainer import ModelTrainer
    
    trainer = ModelTrainer()
    trainer.run(
        data_path=args.data,
        merge=not args.no_merge
    )


def cmd_deploy(args):
    """部署到Ollama"""
    from src.ollama_deployer import OllamaDeployer
    
    deployer = OllamaDeployer()
    
    if args.base_model:
        deployer.deploy_base_model(args.base_model)
    else:
        deployer.run(
            model_path=args.model_path,
            skip_convert=args.skip_convert
        )


# def cmd_run(args):
#     """启动Web应用"""
#     from src.app import run_webapp
#     run_webapp()

def cmd_run(args):
    """启动Web应用"""
    from src.app import run_webapp
    run_webapp()


def cmd_generate(args):
    """命令行生成"""
    from src.generator import NSFCGenerator, ProposalExporter
    
    generator = NSFCGenerator()
    
    if args.literature:
        print("添加文献...")
        generator.add_literature(args.literature)
    
    if args.section == 'all':
        print(f"\n生成完整申请书: {args.topic}")
        results = generator.generate_full_proposal(
            research_topic=args.topic,
            use_literature=not args.no_literature
        )
    else:
        print(f"\n生成 {args.section}: {args.topic}")
        content = generator.generate_section(
            section_type=args.section,
            research_topic=args.topic,
            use_literature=not args.no_literature
        )
        results = {args.section: content}
    
    if args.output:
        ProposalExporter.save(results, args.output, args.topic)
    else:
        for section, content in results.items():
            print(f"\n{'='*50}")
            print(f"{section}")
            print('='*50)
            print(content)


def cmd_add_literature(args):
    """添加文献"""
    from src.literature_manager import LiteratureManager
    
    manager = LiteratureManager(lazy_init=False)
    
    if args.directory:
        print(f"扫描目录: {args.directory}")
        results = manager.add_directory(args.directory)
    elif args.files:
        results = manager.add_files(args.files)
    else:
        print("请指定文献文件或目录")
        print("示例:")
        print("  python main.py add_literature paper1.pdf paper2.docx")
        print("  python main.py add_literature --directory ./papers/")
        return
    
    print(f"\n总计添加: {sum(results.values())} 个文本块")
    
    stats = manager.get_stats()
    print(f"文献库现有: {stats['total_documents']} 篇文献, {stats['total_chunks']} 个文本块")


def cmd_info(args):
    """显示系统信息"""
    from src.config import get_config
    import requests
    
    config = get_config()
    
    print("=" * 50)
    print("国自然写作助手 - 系统信息")
    print("=" * 50)
    
    print(f"\n【配置信息】")
    print(f"  配置文件: {config.config_path}")
    print(f"  基础模型: {config.model.base_model}")
    print(f"  Ollama模型: {config.ollama.model_name}")
    print(f"  Ollama地址: {config.ollama.host}")
    
    # 检查Ollama
    print(f"\n【Ollama状态】")
    try:
        response = requests.get(f"{config.ollama.host}/api/tags", timeout=5)
        if response.status_code == 200:
            models = [m['name'] for m in response.json().get('models', [])]
            print(f"  状态: ✓ 运行中")
            print(f"  可用模型: {', '.join(models) if models else '无'}")
        else:
            print(f"  状态: ✗ 异常")
    except:
        print(f"  状态: ✗ 未运行 (请先执行 'ollama serve')")
    
    # 文献库信息
    print(f"\n【文献库】")
    db_path = config.paths.literature_db
    
    # 检查是否有数据库文件
    has_db = os.path.exists(db_path) and any(
        f.endswith('.sqlite3') or f == 'chroma.sqlite3' 
        for f in os.listdir(db_path)
    ) if os.path.exists(db_path) else False
    
    if has_db:
        try:
            from src.literature_manager import LiteratureManager
            manager = LiteratureManager(lazy_init=False)
            stats = manager.get_stats()
            print(f"  文献数: {stats['total_documents']} 篇")
            print(f"  文本块: {stats['total_chunks']} 个")
        except Exception as e:
            print(f"  状态: 读取失败 - {str(e)}")
    else:
        print(f"  状态: 未初始化")
        print(f"  提示: 使用 'python main.py add_literature 文件' 添加文献")
    
    print(f"\n【支持格式】")
    print(f"  文献: .pdf, .docx, .doc, .md, .markdown, .txt")
    print(f"  训练数据: .md (Markdown)")
    
    print(f"\n【目录说明】")
    print(f"  data/raw/          - 放置训练用的Markdown文件")
    print(f"  data/literature_db/ - 向量数据库存储（自动生成）")
    print(f"  models/finetuned/  - 微调后的模型")
    
    print("=" * 50)


def cmd_clear_literature(args):
    """清空文献库"""
    from src.literature_manager import LiteratureManager
    
    confirm = input("确定要清空文献库吗？(输入 yes 确认): ")
    if confirm.lower() == 'yes':
        manager = LiteratureManager(lazy_init=False)
        manager.clear_all()
        print("文献库已清空")
    else:
        print("已取消")


def main():
    parser = argparse.ArgumentParser(
        description="国自然科学基金申请书写作助手",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  python main.py info                                    # 查看系统信息
  python main.py add_literature paper.pdf                # 添加文献
  python main.py add_literature --directory ./papers/    # 添加目录中所有文献
  python main.py run                                     # 启动Web界面
  python main.py generate -t "研究主题" -s 立项依据       # 命令行生成
  python main.py process_data                            # 处理训练数据
  python main.py train                                   # 微调模型
        """
    )
    
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                       help='配置文件路径')
    
    subparsers = parser.add_subparsers(dest='command', help='可用命令')
    
    # ===== info =====
    subparsers.add_parser('info', help='显示系统信息')
    
    # ===== add_literature =====
    p_lit = subparsers.add_parser('add_literature', help='添加文献到向量库')
    p_lit.add_argument('files', nargs='*', help='文献文件路径')
    p_lit.add_argument('--directory', '-d', type=str, help='文献目录')
    
    # ===== clear_literature =====
    subparsers.add_parser('clear_literature', help='清空文献库')
    
    # ===== run =====
    #subparsers.add_parser('run', help='启动Web应用')
    # ===== run =====
    p_run = subparsers.add_parser('run', help='启动Web应用')

    
    # ===== generate =====
    p_gen = subparsers.add_parser('generate', help='命令行生成')
    p_gen.add_argument('--topic', '-t', type=str, required=True, help='研究主题')
    p_gen.add_argument('--section', '-s', type=str, default='all',
                      choices=['立项依据', '研究内容', '研究方案', 
                              '创新点', '预期成果', '研究基础', 'all'],
                      help='生成的模块')
    p_gen.add_argument('--output', '-o', type=str, help='输出文件路径')
    p_gen.add_argument('--literature', '-l', nargs='*', help='参考文献文件')
    p_gen.add_argument('--no_literature', action='store_true', help='不使用文献检索')
    
    # ===== process_data =====
    p_data = subparsers.add_parser('process_data', help='处理Markdown训练数据')
    p_data.add_argument('--input_dir', type=str, default='./data/raw', help='Markdown文件目录')
    p_data.add_argument('--output', type=str, help='输出文件路径')
    p_data.add_argument('--min_length', type=int, default=100, help='最小内容长度')
    p_data.add_argument('--quality_threshold', type=float, default=0.5, help='质量阈值')
    
    # ===== train =====
    p_train = subparsers.add_parser('train', help='微调模型')
    p_train.add_argument('--data', type=str, help='训练数据路径')
    p_train.add_argument('--no_merge', action='store_true', help='不合并LoRA权重')
    
    # ===== deploy =====
    p_deploy = subparsers.add_parser('deploy', help='部署到Ollama')
    p_deploy.add_argument('--model_path', type=str, help='模型路径')
    p_deploy.add_argument('--skip_convert', action='store_true', help='跳过GGUF转换')
    p_deploy.add_argument('--base-model', type=str, help='使用基础模型')
    
    # 解析参数
    args = parser.parse_args()
    
    # 设置配置
    if args.config:
        from src.config import reload_config
        reload_config(args.config)
    
    # 执行命令
    commands = {
        'info': cmd_info,
        'add_literature': cmd_add_literature,
        'clear_literature': cmd_clear_literature,
        'run': cmd_run,
        'generate': cmd_generate,
        'process_data': cmd_process_data,
        'train': cmd_train,
        'deploy': cmd_deploy,
    }
    
    if args.command in commands:
        commands[args.command](args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()