import os
import subprocess
import requests
import json
import getpass

def setup_github_automation():
    """设置GitHub自动化所需的所有内容"""
    
    # 获取GitHub凭据
    username = input("输入GitHub用户名: ")
    token = getpass.getpass("输入GitHub个人访问令牌: ")
    
    # 保存凭据（安全存储）
    credentials_file = os.path.expanduser("~/.github_credentials")
    with open(credentials_file, "w") as f:
        f.write(json.dumps({"username": username, "token": token}))
    os.chmod(credentials_file, 0o600)  # 只允许用户读写
    
    print("凭据已保存！")
    return username, token

def create_github_repo(repo_name, description="", private=False, username=None, token=None):
    """创建一个新的GitHub仓库"""
    
    # 如果未提供凭据，尝试从文件加载
    if not username or not token:
        credentials_file = os.path.expanduser("~/.github_credentials")
        if os.path.exists(credentials_file):
            with open(credentials_file, "r") as f:
                credentials = json.loads(f.read())
                username = credentials["username"]
                token = credentials["token"]
        else:
            username, token = setup_github_automation()
    
    # 创建API请求
    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github.v3+json"
    }
    
    data = {
        "name": repo_name,
        "description": description,
        "private": private,
        "auto_init": True
    }
    
    response = requests.post("https://api.github.com/user/repos", 
                           headers=headers, 
                           data=json.dumps(data))
    
    if response.status_code == 201:
        print(f"成功创建仓库: {repo_name}")
        return f"https://github.com/{username}/{repo_name}.git"
    else:
        print(f"创建仓库失败: {response.status_code}")
        print(response.json())
        return None

def upload_to_github(local_dir, repo_url=None, repo_name=None, commit_message="自动提交"):
    """自动上传代码到GitHub仓库"""
    
    # 如果未提供repo_url但提供了repo_name，则尝试创建
    if not repo_url and repo_name:
        credentials_file = os.path.expanduser("~/.github_credentials")
        if os.path.exists(credentials_file):
            with open(credentials_file, "r") as f:
                credentials = json.loads(f.read())
                username = credentials["username"]
                token = credentials["token"]
            
            repo_url = create_github_repo(repo_name, username=username, token=token)
        else:
            print("未提供仓库URL，且无法找到凭据来创建仓库")
            return False
    
    # 切换到目标目录
    original_dir = os.getcwd()
    os.chdir(local_dir)
    
    try:
        # 检查是否已存在Git仓库
        is_git_repo = os.path.exists('.git')
        
        if not is_git_repo:
            # 初始化Git仓库
            subprocess.run(["git", "init"], check=True)
            print("初始化新的Git仓库")
        else:
            print("使用已存在的Git仓库")
        
        # 检查是否有可提交的变更
        status = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True, check=True)
        has_changes = bool(status.stdout.strip())
        
        if has_changes:
            # 添加所有文件
            subprocess.run(["git", "add", "."], check=True)
            
            # 提交更改
            subprocess.run(["git", "commit", "-m", commit_message], check=True)
            print("已提交本地更改")
            
            # 获取当前分支名
            branch_cmd = subprocess.run(["git", "branch", "--show-current"], 
                                       capture_output=True, text=True, check=True)
            current_branch = branch_cmd.stdout.strip()
            if not current_branch:  # 如果命令返回空，尝试其他方式获取
                branch_cmd = subprocess.run(["git", "rev-parse", "--abbrev-ref", "HEAD"], 
                                         capture_output=True, text=True, check=True)
                current_branch = branch_cmd.stdout.strip()
            
            if not current_branch:  # 如果还是获取不到，默认为main
                current_branch = "main"
            
            print(f"当前分支: {current_branch}")
            
            # 检查远程仓库是否已经设置
            remote_check = subprocess.run(["git", "remote", "-v"], 
                                      capture_output=True, text=True)
            
            if "origin" not in remote_check.stdout:
                # 添加远程仓库
                subprocess.run(["git", "remote", "add", "origin", repo_url], check=True)
                print(f"添加远程仓库: {repo_url}")
            
            # 推送到GitHub
            subprocess.run(["git", "push", "-u", "origin", current_branch], check=True)
            print(f"代码已成功推送到分支: {current_branch}")
            
        else:
            print("没有变更需要提交")
            
            # 检查本地是否已有提交但未推送的内容
            unpushed = subprocess.run(["git", "log", "@{u}..", "--oneline"], 
                                    capture_output=True, text=True, shell=True)
            
            # 如果返回错误，可能是没有设置上游分支
            if unpushed.returncode != 0:
                # 获取当前分支
                branch_cmd = subprocess.run(["git", "branch", "--show-current"], 
                                         capture_output=True, text=True, check=True)
                current_branch = branch_cmd.stdout.strip() or "main"
                
                # 检查远程仓库是否已经设置
                remote_check = subprocess.run(["git", "remote", "-v"], 
                                          capture_output=True, text=True)
                
                if "origin" not in remote_check.stdout and repo_url:
                    # 添加远程仓库
                    subprocess.run(["git", "remote", "add", "origin", repo_url], check=True)
                    print(f"添加远程仓库: {repo_url}")
                
                # 尝试推送当前分支
                try:
                    subprocess.run(["git", "push", "-u", "origin", current_branch], check=True)
                    print(f"设置上游分支并推送到: {current_branch}")
                except subprocess.CalledProcessError:
                    print(f"无法推送到远程仓库，可能远程分支不存在或存在冲突")
            
        success = True
    except subprocess.CalledProcessError as e:
        print(f"上传过程中出错: {e}")
        success = False
    finally:
        # 恢复原目录
        os.chdir(original_dir)
        return success

# 使用示例
if __name__ == "__main__":
    # 首次设置
    setup_github_automation()  # 第一次运行时取消注释此行
    
    # 创建requirements.txt
    os.system("pip freeze > requirements.txt")
    
    # 创建并上传代码
    project_dir = os.getcwd()  # 当前目录
    repo_name = "linear-regression-demo"  # 仓库名
    
    upload_to_github(
        local_dir=project_dir,
        repo_name=repo_name,
        commit_message="添加线性回归实现和环境配置"
    )
