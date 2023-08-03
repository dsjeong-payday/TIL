# TIL
진정한 자유를 위한 기록
평일 17:00, 주말 random
commit 메시지 포멧: yyyy-mm-dd TIL 루틴

# AI/DL
## NLP Course - Hugging Face
### 1. Transformer Models
#### `pipeline()` - 트랜스포머 모델의 입출력 주고받는 객체 생성
```python
from transformers import pipeline

# "sentiment-analysis" 모델 불러와서 classifier에 할당
classifier = pipeline("sentiment-analysis")

# 위에서 호출한 모델에 inference 시키기
# 내부적으로 (1) 텍스트 전처리 (2) 전처리된 데이터 입력 (3) 모델 예측값 후처리
classifier("I've been waiting for a HuggingFace course my whole life.")  

# [{'label': 'POSITIVE', 'score': 0.9598047137260437}]
```
#### `pipeline()` 인자 - 예시: Zero-shot classification
**zero-shot** 분류기: 파인 튜닝할 필요 없이 바로 사용할 수 있는 분류 모델
```python
from transformers import pipeline

classifier = pipeline("zero-shot-classification")

# candidate_labels = [] 분류
# pipeline() 모델별로 요구 인자 확인할 것
classifier(
    "This is a course about the Transformers library",
    candidate_labels=["education", "politics", "business"],
)

# {'sequence': 'This is a course about the Transformers library',
# 'labels': ['education', 'business', 'politics'],
# 'scores': [0.8445963859558105, 0.111976258456707, 0.043427448719739914]}
```
#### `pipeline()` 특정 모델 호출하기
```python
from transformers import pipeline

generator = pipeline("text-generation", model="distilgpt2")

generator(
    "In this course, we will teach you how to",
    max_length=30,
    num_return_sequences=2,
    # pad_token_id=50256
)
```
그밖에 여러 가지 pre-trained 모델은 [Hugging Face 모델 허브](https://huggingface.co/models)에서 찾아볼 수 있음
#### `pipeline()` 번역 모델 사용해보기
예제 코드
```python
from transformers import pipeline

translator = pipeline("translation", model="Helsinki-NLP/opus-mt-fr-en")
translator("Ce cours est produit par Hugging Face.")

# [{'translation_text': 'This course is produced by Hugging Face.'}]
```

응용 코드 - 한영번역 모델([Helsinki-NLP/opus-mt-ko-en](https://huggingface.co/Helsinki-NLP/opus-mt-ko-en)) 활용
```python
from transformers import pipeline

# https://huggingface.co/Helsinki-NLP/opus-mt-ko-en
translator = pipeline("translation", model="Helsinki-NLP/opus-mt-ko-en")
translator("이 과정은 Hugging Face에서 만들었습니다.")

# [{'translation_text': 'This process was done by Hugging Face.'}]
```

#### `pipeline()` 실습
여러 모델을 순차적으로 불러와서 inference 시켜봄
(1) 번역 모델: 뉴스 기사 제목 -> 영어로 번역
(2) 분류기: 영어로 번역된 기사 제목을 5대 분류(정치, 경제, 사회, 문화, 세계)에 넣기. 
(3) 생성기: 기사 제목을 넣어서 내용 추가하기
(4) [기사 분류] 제목\n본문 형태로 출력
```python
from transformers import pipeline

translator = pipeline("translation", model="Helsinki-NLP/opus-mt-ko-en")
classifier = pipeline("zero-shot-classification")
generator = pipeline("text-generation", model="distilgpt2")
labels = ["politics", "business", "society", "culture", "global"]

if __name__ == '__main__':
    title = input("기사 제목을 입력해주세요: ")
    eng_title = translator(title)[0]['translation_text']
    print(f"번역된 기사 제목: {eng_title}")

    classified = classifier(eng_title, candidate_labels=labels)
    highest_label = classified['labels'][0]

    article = generator(eng_title, max_new_tokens=30, pad_token_id=50256)[0]['generated_text']

    print(f"[{highest_label}] {article}")
```

```
기사 제목을 입력해주세요: 챗GPT 주춤하자 분주해진 경쟁사들
번역된 기사 제목: Let's get the GGPT running, and we'll have busy competitors.
[business] Let's get the GGPT running, and we'll have busy competitors. We will look and say 'OK that's done!' and then we'll have a round of tournaments starting in November. You'll be able to play

```
```
기사 제목을 입력해주세요: 또 '신림역 살인 예고'... 경찰, 7번째 예고글 추적
번역된 기사 제목: Police, we're tracking for the seventh one.
[business] Police, we're tracking for the seventh one.
We are also tracking you from around the world for any issues when the app is installed and your credit card is not
```

```
기사 제목을 입력해주세요: 서울 관악구 신림동 일대에서 살인을 예고하는 글이 또 온라인에 올라와 경찰이 수사에 나섰다.
번역된 기사 제목: Police have come online to investigate the case.
[global] Police have come online to investigate the case. But they say they are still awaiting an inquest into the deaths of two boys from gang violence, including 15-year-old Jody Bostak
```



# Docker
## Docker Desktop Tutorial
### What is a container?
Containers: 격리된 환경
- Name 
	- logs
	- inspect
	- terminal
	- files
	- status
- Image
- Status
- Port(s)
- Last started
- Actions (run/stop, others)
- Delete
### How do I run a container?
`Dockerfile` & `Codes` -> `image` -> `container`.
아래와 같이 소스코드와 `Dockerfile`를 작성하고 `docker build`를 실행한다. 빌드가 끝나면 Docker `image`가 생성된다. `image`를 실행하면 `container`를 생성하고 실행한다.
```Dockerfile
# Start your image with a node base image
FROM node:18-alpine

# The /app directory should act as the main application directory
WORKDIR /app

# Copy the app package and package-lock.json file
COPY package*.json ./

# Copy local directories to the current local directory of our docker image (/app)
COPY ./src ./src
COPY ./public ./public

# Install node packages, install serve, build the app, and remove dependencies at the end
RUN npm install \
&& npm install -g serve \
&& npm run build \
&& rm -fr node_modules

EXPOSE 3000

# Start the app using serve command
CMD [ "serve", "-s", "build" ]
```
### Multi-container applications
여러 컨테이너끼리 서로 통신하는 앱 만들기. (`ExpressJS` & `Node`, `MongoDB`) `compose.yaml` 파일에 여러 서비스를 정의한다.
```yaml
# Comments are provided throughout this file to help you get started.
# If you need more help, visit the Docker compose reference guide at
# https://docs.docker.com/compose/compose-file/

# Here the instructions define your application as two services called "todo-app" and “todo-database”
# The service “todo-app” is built from the Dockerfile in the /app directory,
# and the service “todo-database” uses the official MongoDB image
# from Docker Hub - https://hub.docker.com/_/mongo.
# You can add other services your application may depend on here.
services:
  todo-app:
    build:
      context: ./app
    links:
      - todo-database
    environment:
      NODE_ENV: production
    ports:
      - 3000:3000

    todo-database:
      image: mongo:6
      #volumes:
      #  - database:/data/db
      ports:
        - 27017:27017

#volumes:
  #database:
```
`services` 항목 아래에 `todo-app`과 `todo-database`라는 서비스가 각각 정의되어 있다. `docker compose` 실행하면 `compose.yaml`에 정의된 모든 서비스를 빌드하고 실행한다. `containers` UI에서는 앱 스택으로 나타난다.
### Persist your data between containers
`container`는 격리된 파일 시스템이지만, 로컬의 다른 환경이나 다른 `container`와 파일을 공유해야 할 때도 있다. (예: Database file)  이때 `volume`을 활용한다. `volume`이란 로컬 파일 시스템에 존재하지만 `Docker`에 의해 관리되는 공간이다. 위의 `compose.yaml`에서 주석 처리를 아래와 같이 푼다.
```yaml
# todo-database 서비스 - database라는 volume의 특정 경로를 활용
  todo-database:
    image: mongo:6
    volumes:
      - database:/data/db
    ports:
      - 27017:27017

...

# 여러 volume 중 database는 아무 서비스나 활용 가능
volumes:
  database:
```
### Containerize your application
`docker init`: `Dockerfile`, `compose.yaml` 파일 등을 자동으로 구성 및 생성(`build`). 빌드가 끝나면 `docker compose up -d`로 실행 가능
### Run Docker Hub Images
`Docker Desktop`에서 `Ctrl + K`  누르면 `Docker Hub` 검색창 입력 가능. 원하는 이미지를 선택하고 `Run` 하거나, `View on Hub`로 상세 정보 확인. `Containers` 화면에서 선택한 이미지로 만들어진 `container` 확인 가능.
### Publish your image
로컬에 저장된 특정 이미지의 이름을 수정하는 코드. [YOUR-USERNAME]은 실제 계정명으로 바꿀 것.
```
docker tag docker/welcome-to-docker [YOUR-USERNAME]/welcome-to-docker
```
`images` 화면에서 해당 이미지의 `Actions` 클릭, `Push to Hub` 클릭하면 내 `Docker Hub` 저장소에 올라감. 
## Docker Python Application Guide
### Build images
### Run your image as a container
### Use containers for development
#### 목표 - 로컬 개발환경 설치하기.
#### 컨테이너에서 DB 실행하기
`volume`과 `network`를 설정함으로써 데이터를 유지시키고 앱과 DB 간의 통신을 설정함. 그리고나서 `compose` 파일로 한 번에 실행할 예정. 
```
docker volume create mysql
docker volume create mysql_config
docker network create mysqlnet
```
위는 MySQL 데이터 볼륨, MySQL 설정파일 볼륨, 그리고 네트워크를 생성하는 명령어임. 이어서 아래와 같이 컨테이너를 실행함.
```
docker run --rm -d -v mysql:/var/lib/mysql \
  -v mysql_config:/etc/mysql -p 3306:3306 \
  --network mysqlnet \
  --name mysqldb \
  -e MYSQL_ROOT_PASSWORD=p@ssw0rd1 \
  mysql
```
컨테이너가 실행되면 `docker exec -ti mysqldb mysql -u root -p`로 MySQL 접속 가능. 해당 컨테이너에 `mysql -u root -p`의 명령어를 전달하는 명령어임.
#### 앱을 DB에 연결하기
아래는 기능이 추가된 Python 앱 코드임. `/initdb`에서 데이터베이스와 테이블을 초기화하며, `/widgets`에서는 위젯을 불러옴.
```py
import mysql.connector
import json
from flask import Flask

app = Flask(__name__)

@app.route('/')
def hello_world():
    return 'Hello, Docker!'

@app.route('/widgets')
def get_widgets():
    mydb = mysql.connector.connect(
        host="mysqldb",
        user="root",
        password="p@ssw0rd1",
        database="inventory"
    )
    cursor = mydb.cursor()


    cursor.execute("SELECT * FROM widgets")

    row_headers=[x[0] for x in cursor.description] #this will extract row headers

    results = cursor.fetchall()
    json_data=[]
    for result in results:
        json_data.append(dict(zip(row_headers,result)))

    cursor.close()

    return json.dumps(json_data)

@app.route('/initdb')
def db_init():
    mydb = mysql.connector.connect(
        host="mysqldb",
        user="root",
        password="p@ssw0rd1"
    )
    cursor = mydb.cursor()

    cursor.execute("DROP DATABASE IF EXISTS inventory")
    cursor.execute("CREATE DATABASE inventory")
    cursor.execute("USE inventory")

    cursor.execute("DROP TABLE IF EXISTS widgets")
    cursor.execute("CREATE TABLE widgets (name VARCHAR(255), description VARCHAR(255))")
    cursor.close()

    return 'init database'

if __name__ == "__main__":
    app.run(host ='0.0.0.0')
```
아래와 같이 `mysql-connector-python` 모듈을 설치하고 `requirements.txt` 목록에도 추가.	
```
pip3 install mysql-connector-python
pip3 freeze | grep mysql-connector-python >> requirements.txt
```
이어서 `image` 빌드하기.
```
docker build --tag python-docker-dev .
```
앞서 실행한 DB 컨테이너에서 `network`에 연결한 것처럼, Python 앱도 동일한 `network`에 연결함.
```
docker run \
  --rm -d \
  --network mysqlnet \
  --name rest-server \
  -p 8000:5000 \
  python-docker-dev
```
end-point 테스트 해보기. 아직 데이터가 없기 때문에 `/widgets`는 빈 JSON을 반환함.
```
curl http://localhost:8000/initdb
curl http://localhost:8000/widgets
```
#### Compose 이용하여 로컬 개발 환경 설치
`docker-compose.dev.yml`
```yml
version: '3.8'

services:
 web:
  build:
   context: .
  ports:
  - 8000:5000
  volumes:
  - ./:/app

 mysqldb:
  image: mysql
  ports:
  - 3306:3306
  environment:
  - MYSQL_ROOT_PASSWORD=p@ssw0rd1
  volumes:
  - mysql:/var/lib/mysql
  - mysql_config:/etc/mysql

volumes:
  mysql:
  mysql_config:
```
앞서 `docker run`을 실행할 때 추가적으로 입력한 여러 가지 플래그를 `Compose` 파일에 선언할 수 있음. 로컬 파일 경로를 컨테이너에 맵핑하였기 때문에, 로컬에서 수정한 내용이 컨테이너에도 반영됨. `network`에 관한 설정은 생략되었는데, 동일한 `Compose` 파일이 자동으로 생성하고 서비스들을 연결하기 때문임. 
기존의 컨테이너를 종료하고 `docker compose` 실행
```
docker compose -f docker-compose.dev.yml up --build
```
`--build` 플래그를 넣었기 때문에, 이미지를 컴파일하고 컨테이너를 시작할 것임. 컨테이너가 실행되고 나서 `curl` 테스트를 해보면 이전과 동일한 결과 나옴.
### Configure CI/CD
1. GitHub 저장소 만들기
	- `Settings` -> `Secrets and variables` -> `Actions`
	- `New repository secret` 기능으로 다음 두 가지 시크릿 만들기
		- `DOCKERHUB_USERNAME`
		- `DOCKERHUB_TOKEN`: Docker Hub에 생성한 접근 토큰
	- Docker Hub 계정 접근 토큰 만들기: `Account Settings` -> `Security` -> `New Access Token`
2. GitHub Actions workflow 정의하기
	- `Actions` -> `set up a workflow yourself` -> `.github/workflows/main.yml` 수정하기
	```yml
	name: ci

	on:
	  push:
	    branches:
	      - "main"

	jobs:
	  build:
	    runs-on: ubuntu-latest
	```
	`on.push.branches`: branch 특정하면 해당 branch에 push event 발생할 때마다 workflow 작동

	```yml
	jobs:
	  build:
	    runs-on: ubuntu-latest
	    steps:
	      -
	        name: Checkout
	        uses: actions/checkout@v3
	      -
	        name: Login to Docker Hub
	        uses: docker/login-action@v2
	        with:
	          username: ${{ secrets.DOCKERHUB_USERNAME }}
	          password: ${{ secrets.DOCKERHUB_TOKEN }}
	      -
	        name: Set up Docker Buildx
	        uses: docker/setup-buildx-action@v2
	      -
	        name: Build and push
	        uses: docker/build-push-action@v4
	        with:
	          context: .
	          file: ./Dockerfile
	          push: true
	          tags: ${{ secrets.DOCKERHUB_USERNAME}}/clockbox:latest
	```
3. Workflow 실행하기
	- 위에서 추가한 `main.yml` 커밋하기. `Commit changes` 
	- `Actions` 탭으로 이동하면 현재 작동 중인 workflow 나옴
	- Docker Hub의 저장소 메뉴로 가보면 새 저장소가 추가되어 있음. GitHub Actions을 통해 Docker image가 push된 것임.
GitHub 저장소를 로컬에 내려받고 파일을 수정한다. 수정 사항을 커밋하면 workflow에 의해 Docker Hub에도 자동으로 push 된다.
### Deploy your app
Docker `container`를 배포하려면 Azure ACI, AWS ECS와 같이 클라우드 서비스에서 지원하는 컨테이너 전용 서비스를 활용하거나, Kubernetes 서버를 구축할 수도 있다.

# 여러 가지 생산성 툴
## Vim - Text Editor
### 단축키
#### 이동
- `h`, `j`, `k`, `l`: 각각 왼쪽, 위, 아래, 오른쪽
- `w`, `b`: 바로 다음, 이전의 단어
- `W`, `B`: 바로 다음, 이전의 단어(공백 기준)
- `e`, `ge`: 바로 다음, 이전 단어의 끝
#### 편집
- `i`: `i`nsert 모드로 전환
- `a`: `insert` 모드로 전환 `a`fter the cursor.
- `I`, `A`: (1) `insert` 모드로 전환, (2) 커서를 문장의 맨 앞이나 뒤로 이동
- `o`, `O`: (1) 아래(`o`)나 위(`O`)에 빈 줄 추가, (2) `insert` 모드로 전환
- `ESC`, `Ctrl+[`: `insert` 모드 종료하기
- `d`: (선택된 부분) 삭제하기 
- `dd`: 줄 삭제하기
- `c`: (선택된 부분) 삭제하고 `insert` 모드로 전환 - `cc`: 줄 삭제하고 `insert` 모드로 전환
#### operators
(visual 모드에서도 작동함)
- `d`: `visual` 모드에서 커서 위치부터 이동하는 방향으로 삭제
- `c`: `visual` 모드에서 `d`처럼 삭제하고 `insert` 모드로 전환
- `y`: 커서 위치부터 이동하는 방향으로 복사
- `>`, `<`: 들여쓰기 넣기, 빼기
#### Visual 모드
- `v`: `visual` 모드로 전환
- `V`: linewise visual 모드
- `Ctrl+v`: visual block 모드
- `ESC`, `Ctrl+[`: `visual` 모드 종료하기

## Notion
### 데이터베이스
#### 하위 작업 유형
데이터베이스 우측 상단의 `...` 버튼 클릭
하위 항목 `꺼짐` -> `하위 항목 켜기`
기존 항목에 드랍다운(▶ -> ▼) 클릭하고 하위 항목 추가하기


## markdown 편집
### [StackEdit](https://stackedit.io/)
> In-browser Markdown editor
#### 주요 기능
- md 작성
- WYSIWYG controls (화면 상단의 에디터 패널)
- 레이아웃 사용자화, 실시간 스크롤 연동
- 외부 저장소 연동(Github, Google Drive, Dropbox 등)
- 협업, 댓글
- Desktop 앱 -> 오프라인 사용 가능
- 확장 기능: Github 스타일 markdown, LaTeX 수식 표현, UML 다이어그램, 악보, 이모지
### [Mermaid](https://mermaid.js.org/)
> JavaScript based diagramming and charting tool that renders Markdown-inspired text definitions to create and modify diagrams dynamically.

특정 문법(Markdown과 유사)을 따르는 텍스트 ---렌더링---> 다이어그램, 차트 생성/수정

#### 사용방법
- [공식 웹 에디터](https://mermaid.live/)
- [플러그인](https://mermaid.js.org/ecosystem/integrations.html)
- JavaScript API 호출
- (NodeJS 프로젝트에) 의존성 배포
# 대한민국 법령, Compliance
## 「개인정보보호법」
#### 법 제26조(업무위탁에 따른 개인정보의 처리 제한)
- `법 제26조`는 개인정보처리자(=위탁자)가 제3자(=수탁자)에게 개인정보 처리 업무를 위탁할 때 그 범위와 방법 등을 제한함
- 또한 업무 위탁에 따른 추가적인 의무를 부과함(제2항 - 정보주체에게 수탁자 공개, 제3항 - 정보주체에게 위탁 업무 내용 고지, 제4항 - 수탁자 교육 및 관리감독)
- 제5항(업무 범위 초과 이용 혹은 제3자 제공), 제6항(재위탁 시 위탁자의 동의 필요), 제8항(개인정보처리자로서 준용 조항)은 수탁자의 의무
- 제7항: 손해배상책임에 대한 수탁자의 지위 = 위탁자의 소속 직원
- 특히 `제26조제5항(업무 범위 초과 이용 혹은 제3자 제공)`은 같은 `법 제71조 제2호`에 따라 위반 시 5년 이하의 징역 또는 5천만원 이하의 벌금
```
제26조(업무위탁에 따른 개인정보의 처리 제한) ① 개인정보처리자가 제3자에게 개인정보의 처리 업무를 위탁하는 경우에는 다음 각 호의 내용이 포함된 문서로 하여야 한다. <개정 2023. 3. 14.>
1. 위탁업무 수행 목적 외 개인정보의 처리 금지에 관한 사항
2. 개인정보의 기술적ㆍ관리적 보호조치에 관한 사항
3. 그 밖에 개인정보의 안전한 관리를 위하여 대통령령으로 정한 사항
② 제1항에 따라 개인정보의 처리 업무를 위탁하는 개인정보처리자(이하 “위탁자”라 한다)는 위탁하는 업무의 내용과 개인정보 처리 업무를 위탁받아 처리하는 자(개인정보 처리 업무를 위탁받아 처리하는 자로부터 위탁받은 업무를 다시 위탁받은 제3자를 포함하며, 이하 “수탁자”라 한다)를 정보주체가 언제든지 쉽게 확인할 수 있도록 대통령령으로 정하는 방법에 따라 공개하여야 한다. <개정 2023. 3. 14.>
③ 위탁자가 재화 또는 서비스를 홍보하거나 판매를 권유하는 업무를 위탁하는 경우에는 대통령령으로 정하는 방법에 따라 위탁하는 업무의 내용과 수탁자를 정보주체에게 알려야 한다. 위탁하는 업무의 내용이나 수탁자가 변경된 경우에도 또한 같다.
④ 위탁자는 업무 위탁으로 인하여 정보주체의 개인정보가 분실ㆍ도난ㆍ유출ㆍ위조ㆍ변조 또는 훼손되지 아니하도록 수탁자를 교육하고, 처리 현황 점검 등 대통령령으로 정하는 바에 따라 수탁자가 개인정보를 안전하게 처리하는지를 감독하여야 한다. <개정 2015. 7. 24.>
⑤ 수탁자는 개인정보처리자로부터 위탁받은 해당 업무 범위를 초과하여 개인정보를 이용하거나 제3자에게 제공하여서는 아니 된다.
⑥ 수탁자는 위탁받은 개인정보의 처리 업무를 제3자에게 다시 위탁하려는 경우에는 위탁자의 동의를 받아야 한다. <신설 2023. 3. 14.>
⑦ 수탁자가 위탁받은 업무와 관련하여 개인정보를 처리하는 과정에서 이 법을 위반하여 발생한 손해배상책임에 대하여는 수탁자를 개인정보처리자의 소속 직원으로 본다. <개정 2023. 3. 14.>
⑧ 수탁자에 관하여는 제15조부터 제18조까지, 제21조, 제22조, 제22조의2, 제23조, 제24조, 제24조의2, 제25조, 제25조의2, 제27조, 제28조, 제28조의2부터 제28조의5까지, 제28조의7부터 제28조의11까지, 제29조, 제30조, 제30조의2, 제31조, 제33조, 제34조, 제34조의2, 제35조, 제35조의2, 제36조, 제37조, 제37조의2, 제38조, 제59조, 제63조, 제63조의2 및 제64조의2를 준용한다. 이 경우 “개인정보처리자”는 “수탁자”로 본다. <개정 2023. 3. 14.>
```

## 「정보통신망법」
#### 정보보호 최고책임자(CISO)
- 참고자료: `정보보호 최고책임자 지정신고 제도 안내서(과기부, KISA; 2021)`
#### 법 제45조의3, 시행령 제36조의7
- `법 제45조의3`은 정보보호 최고책임자 지정 의무(+ 예외), 지정 방법 및 절차, 겸직 제한, 업무, 자격요건 등에 대해 기술함
```
제45조의3(정보보호 최고책임자의 지정 등) ① 정보통신서비스 제공자는 정보통신시스템 등에 대한 보안 및 정보의 안전한 관리를 위하여 대통령령으로 정하는 기준에 해당하는 임직원을 정보보호 최고책임자로 지정하고 과학기술정보통신부장관에게 신고하여야 한다. 다만, 자산총액, 매출액 등이 대통령령으로 정하는 기준에 해당하는 정보통신서비스 제공자의 경우에는 정보보호 최고책임자를 신고하지 아니할 수 있다. <개정 2014. 5. 28., 2017. 7. 26., 2018. 6. 12., 2021. 6. 8.>
② 제1항에 따른 신고의 방법 및 절차 등에 대해서는 대통령령으로 정한다. <신설 2014. 5. 28.>
③ 제1항 본문에 따라 지정 및 신고된 정보보호 최고책임자(자산총액, 매출액 등 대통령령으로 정하는 기준에 해당하는 정보통신서비스 제공자의 경우로 한정한다)는 제4항의 업무 외의 다른 업무를 겸직할 수 없다. <신설 2018. 6. 12.>

④ 정보보호 최고책임자의 업무는 다음 각 호와 같다. <개정 2021. 6. 8.>
1. 정보보호 최고책임자는 다음 각 목의 업무를 총괄한다.
가. 정보보호 계획의 수립ㆍ시행 및 개선
나. 정보보호 실태와 관행의 정기적인 감사 및 개선
다. 정보보호 위험의 식별 평가 및 정보보호 대책 마련
라. 정보보호 교육과 모의 훈련 계획의 수립 및 시행

2. 정보보호 최고책임자는 다음 각 목의 업무를 겸할 수 있다.
가. 「정보보호산업의 진흥에 관한 법률」 제13조에 따른 정보보호 공시에 관한 업무
나. 「정보통신기반 보호법」 제5조제5항에 따른 정보보호책임자의 업무
다. 「전자금융거래법」 제21조의2제4항에 따른 정보보호최고책임자의 업무
라. 「개인정보 보호법」 제31조제2항에 따른 개인정보 보호책임자의 업무
마. 그 밖에 이 법 또는 관계 법령에 따라 정보보호를 위하여 필요한 조치의 이행

⑤ 정보통신서비스 제공자는 침해사고에 대한 공동 예방 및 대응, 필요한 정보의 교류, 그 밖에 대통령령으로 정하는 공동의 사업을 수행하기 위하여 제1항에 따른 정보보호 최고책임자를 구성원으로 하는 정보보호 최고책임자 협의회를 구성ㆍ운영할 수 있다. <개정 2014. 5. 28., 2018. 6. 12.>
⑥ 정부는 제5항에 따른 정보보호 최고책임자 협의회의 활동에 필요한 경비의 전부 또는 일부를 지원할 수 있다. <개정 2014. 5. 28., 2015. 6. 22., 2018. 6. 12.>
⑦ 정보보호 최고책임자의 자격요건 등에 필요한 사항은 대통령령으로 정한다. <신설 2018. 6. 12.>

[본조신설 2012. 2. 17.]
```
- 정보통신서비스 제공자 중 일부는 CISO 지정 및 신고 의무가 없다. 대신 사업주 또는 대표자를 CISO로 지정한 것으로 본다. (아래는 시행령)
```
제36조의7(정보보호 최고책임자의 지정 및 겸직금지 등) ① 법 제45조의3제1항 본문에서 “대통령령으로 정하는 기준에 해당하는 임직원”이란 다음 각 호의 구분에 따른 사람을 말한다. <신설 2021. 12. 7.>

1. 다음 각 목의 어느 하나에 해당하는 정보통신서비스 제공자: 사업주 또는 대표자
가. 자본금이 1억원 이하인 자
나. 「중소기업기본법」 제2조제2항에 따른 소기업
다. 「중소기업기본법」 제2조제2항에 따른 중기업으로서 다음의 어느 하나에 해당하지 않는 자
1) 「전기통신사업법」에 따른 전기통신사업자
2) 법 제47조제2항에 따라 정보보호 관리체계 인증을 받아야 하는 자
3) 「개인정보 보호법」 제30조제2항에 따라 개인정보 처리방침을 공개해야 하는 개인정보처리자
4) 「전자상거래 등에서의 소비자보호에 관한 법률」 제12조에 따라 신고를 해야 하는 통신판매업자
...
(중략)
...
② 법 제45조의3제1항 단서에서 “자산총액, 매출액 등이 대통령령으로 정하는 기준에 해당하는 정보통신서비스 제공자”란 정보통신서비스 제공자로서 제1항제1호 각 목의 어느 하나에 해당하는 자를 말한다. <개정 2021. 12. 7.>
③ 법 제45조의3제1항 단서에 해당하는 자가 정보보호 최고책임자를 신고하지 않은 경우에는 사업주나 대표자를 정보보호 최고책임자로 지정한 것으로 본다. <신설 2021. 12. 7.>
```
- 「중소기업기본법」상 `중기업` 중 개인정보 처리방침 공개 의무자는 사업주/대표, 이사, 혹은 `정보보호 관련 업무를 총괄하는 부서의 장`을 CISO로 지정해야 한다. 그런데 CISO는 아무나 지정할 수 없다. (아래는 시행령)
```
④ 법 제45조의3제1항 및 제7항에 따라 정보통신서비스 제공자가 지정ㆍ신고해야 하는 정보보호 최고책임자는 다음 각 호의 어느 하나에 해당하는 자격을 갖추어야 한다. 이 경우 정보보호 또는 정보기술 분야의 학위는 「고등교육법」 제2조 각 호의 학교에서 「전자금융거래법 시행령」 별표 1 비고 제1호 각 목에 따른 학과의 과정을 이수하고 졸업하거나 그 밖의 관계법령에 따라 이와 같은 수준 이상으로 인정되는 학위로 하고, 정보보호 또는 정보기술 분야의 업무는 같은 비고 제3호 및 제4호에 따른 업무로 한다. <개정 2021. 12. 7., 2022. 8. 9.>

1. 정보보호 또는 정보기술 분야의 국내 또는 외국의 석사학위 이상 학위를 취득한 사람
2. 정보보호 또는 정보기술 분야의 국내 또는 외국의 학사학위를 취득한 사람으로서 정보보호 또는 정보기술 분야의 업무를 3년 이상 수행한 경력(학위 취득 전의 경력을 포함한다)이 있는 사람
3. 정보보호 또는 정보기술 분야의 국내 또는 외국의 전문학사학위를 취득한 사람으로서 정보보호 또는 정보기술 분야의 업무를 5년 이상 수행한 경력(학위 취득 전의 경력을 포함한다)이 있는 사람
4. 정보보호 또는 정보기술 분야의 업무를 10년 이상 수행한 경력이 있는 사람
5. 법 제47조제6항제5호에 따른 정보보호 관리체계 인증심사원의 자격을 취득한 사람
6. 해당 정보통신서비스 제공자의 소속인 정보보호 관련 업무를 담당하는 부서의 장으로 1년 이상 근무한 경력이 있는 사람
```
- CISO는 개인정보보호 책임자(CPO)를 겸할 수 있다. (`라. 「개인정보 보호법」 제31조제2항에 따른 개인정보 보호책임자의 업무`) 물론 별도로 지정할 수도 있다. 단, CPO가 CISO를 겸임할 수는 없다.
## 「소득세법」
#### 제134조(근로소득에 대한 원천징수시기 및 방법)
#### 제137조(근로소득세액의 연말정산)
#### 제140조(근로소득자의 소득공제 등 신고)
- 제137조에 따라 연말정산을 할 때 근로소득자가 종합소득공제 및 세액공제를 적용받으려는 경우 ... 2월분의 근로소득을 받기 전까지 "`근로소득자 소득ㆍ세액 공제신고서`" 제출해야
- `시행령 제198조(근로소득자의 소득공제 및 세액공제신고)`: 제140조제1항에 대한 상세 방법
    - ...소득ㆍ세액 공제신고서를 원천징수의무자에게 제출(국세정보통신망에 의한 제출을 포함한다)하여야...
    - 제2항: 소득ㆍ세액 공제신고서에 주민등록표등본을 첨부하여 제출하여야 ... 공제대상 배우자 또는 부양가족이 변동되지 아니한 때에는 주민등록표등본을 제출하지 아니
    - 제3항: 제1항에 따른 신고서를 제출함에 있어서 법 제53조제2항(거주자 또는 동거가족(직계비속ㆍ입양자는 제외한다) 본래의 주소 또는 거소에서 일시 퇴거한 경우)에 해당하는 자가 있는 경우에는 일시퇴거자 동거가족상황표를 근로소득자 소득ㆍ세액 공제신고서에 첨부하여야 한다.
## 4대보험
- 근거 법령: 「국민연금법」, 「국민건강보험법」, 「고용보험법」, 「산업재해보상보험법」

# 웹 개발
## 웹 기술
### RESTful API
- REST (Representational State Transfer): 소프트웨어 아키텍처 양식. 아래의 제약조건을 충족함으로써 경량, 유지보수성, 확장성 지향 -> 클라우드 기반이나 모바일 앱에 적합
    1. 자원(Resources): 시스템에 존재하는 각각의 개체(entity)는 유일한 URI(Uniform Resource Identifier)를 통해 접근 가능. text 파일, HTML 페이지, 이미지, 비디오, 비즈니스 데이터 등
    2. 상태없음(Stateless): 클라이언트로부터 서버로 향하는 각각의 요청은 반드시 그 요청을 수행하는 데에 필요한 모든 정보를 갖추어야 함. 서버는 가장 최근의 HTTP 요청에 대해 아무것도 저장하지 않아야 함. 각각의 요청은 서로 독립적으로 처리됨.
    3. Client-Server Architecture: 클라이언트는 UI/UX를 담당하며, 서버는 요청을 처리하고 자원을 관리하는 역할을 담당함. 클라이언트와 서버는 서로 독립적으로 가동되며, 개발과 개선 또한 분리하여 이루어짐.
    4. Cacheable: 클라이언트가 HTTP 응답을 캐싱할 수 있음. 추후 해당 정보가 필요할 때 새로운 요청을 보내는 대신 캐싱된 데이터를 참조함.
    5. Uniform Interface: 클라이언트와 서버 간의 통신 방식을 몇 가지 표준화된 방식으로 정의함. (GET, POST, PUT, DELETE, ...)
    6. Layered System: 계층화된 형태의 애플리케이션을 허용함. 각각의 계층마다 역할과 책임 부여. 확장성과 모듈성.
    7. Code on Demand (선택): 서버가 클라이언트에게 실행 가능한 코드를 전달함으로써 일시적으로 기능을 확장/사용자화 할 수 있음.
- RESTful API(Application Programming Interface): 위의 REST 제약조건을 만족하는 HTTP 서비스
## mdn web docs
### 서버측 웹사이트 프로그래밍(웹서버 개발)
- [참고](https://developer.mozilla.org/en-US/docs/Learn/Server-side)

#### Introduction -
- web server - `HTTP` - web browser (client)
- `HTTP` (HyperText Trnasfer Protocol): 요청 -> 응답
- 웹 서버: HTTP 요청이 들어올 때까지 대기, 요청이 들어오면 처리 및 응답 반환

##### 정적 사이트(static)
- 항상 일정하게 하드코딩된 컨텐츠 제공
- server-side: 파일 -> 웹 서버
- client-side: 웹 서버 <-> 브라우저
- 미리 만들어진 파일: HTML, CSS, Javascript, 그 외

##### 동적 사이트(dynamic)
- 이용자 요구에 따라 웹서비스 자원을 생성하여 제공
- 주로 Database에서 얻은 데이터를 HTML 템플릿의 placeholder에 전달
- **server-side programming**
- server-side
    - Web Server
    - Files: 정적 자원 - CSS, JavaScript, images, 그 외
    - Web Application
    - Database
- client-side: HTML, CSS, JavaScript -> 브라우저

##### server-side vs. client-side 프로그래밍
- Client-side: 브라우저에서 작동하는 코드. 주로 웹페이지의 모습, 행동(UI 요소 스타일, 레이아웃, 네비게이션, 양식 등)
- Server-side: 요청에 대하여 어느 내용물(*which contents*)을 보여줄 것인지 결정. (데이터 처리 및 검증, 데이터베이스와 통신, 클라이언트에게 보여줄 데이터 발송 등)
    - 프로그래밍 언어: PHP, Python, Ruby, C#, JavaScript(NodeJS) 등 개발자가 고를 수 있음
    - 서버의 OS에 접근 가능
- **웹 프레임워크**: client-side 혹은 server-side의 각자 개발 목적이 다르기에 프레임워크가 제공하는 요소와 기능도 서로 다름

##### server-side 개발을 통해 할 수 있는 것들
- 효율적인 정보 저장과 전달
- 사용자화된 UX
- 컨텐츠에 대한 접근권한 통제
- 세션/상태 정보 저장
- 알림과 소통
- 이용자 데이터 분석

## HTTP
- HyperText Transfer Protocol

### 주요 메소드
#### GET
#### POST
- 주로 HTML form을 통해 전달 -> 서버에 변경사항
- 헤더 `Content-type`: 요청 본문의 유형 특정
    - `text/plain`
    - `application/json` 등
- 헤더 `Authorization`: `POST` 요청을 보내는 주체의 권한 소유 여부 판별

## json
- JSON (JavaScript Object Notation)
- 데이터 형식
- 여러 웹 기반 서비스에서 데이터를 주고 받을 때 활용

# 정보보호/보안
## 취약점 진단/분석
- 법적 근거: `「개인정보의 안전성 확보조치 기준」 제6조(접근통제)` ④ 고유식별정보를 처리하는 개인정보처리자는 인터넷 홈페이지를 통해 고유식별정보가 유출ㆍ변조ㆍ훼손되지 않도록 연 1회 이상 취약점을 점검하고 필요한 보완 조치를 하여야 한다.


### [OWASP Top 10: 2021](https://owasp.org/Top10/)
OWASP에서 선정한 웹 애플리케이션 10대 취약점. 최근 개정판은 2021년에 2017년 목록을 수정한 것이다.
#### [ A01:2021-Broken Access Control](https://owasp.org/Top10/A01_2021-Broken_Access_Control/) - 접근 통제 실패
웹 앱은 이용자에게 여러 가지 행동 권한을 부여한다. 권한 통제에 실패하면 이용자는 허가받지 않은 행동을 취할 수 있게 된다. 예를 들면 인가받지 않은 자료에 대한 조회/수정/삭제하거나, 혹은 허용 범위를 초과한 기능을 실행할 수도 있다. 구체적인 취약점 예시는 다음과 같다.

- 최소 권한의 원칙, 거부 기본값 원칙 미준수: 특정 기능, 역할, 이용자에게 허가되어야 할 권한이 아무나에게 제공
- 권한 통제 확인 우회: URL 수정, 내부 앱 상태, HTML 페이지, 혹은 API 요청을 조적함으로써 권한 확인을 우회할 수 있는 지점
- (안전하지 않은 direct object 참조) 고유 식별자를 제공함으로써 다른 사람의 계정을 보거나 수정할 수 있게 허용
- 권한 통제가 누락된 HTTP methods(`POST`, `PUT`, 그리고 `DELETE`)에 API 접근
- 권한 상승 허용: 관리자 권한이 있거나 관리자 계정으로 로그인 한 것처럼 속이는 경우
- 메타정보 조작: `JSON Web Token (JWT)`, 쿠키, 혹은 숨겨진 필드 등을 탈취하거나 악용
- CORS 설정 오류: 허가받지 않은/신뢰하지 않는 출처로부터 API 접근 허용
- 인증이 필요하거나 제한된 페이지를 그렇지 않은 이용자에게 강제로 보여주는 경우

#### [A02:2021-Cryptographic Failures](https://owasp.org/Top10/A02_2021-Cryptographic_Failures/)
#### [A03:2021-Injection](https://owasp.org/Top10/A03_2021-Injection/)
#### [A04:2021-Insecure Design](https://owasp.org/Top10/A04_2021-Insecure_Design/)
#### [A05:2021-Security Misconfiguration](https://owasp.org/Top10/A05_2021-Security_Misconfiguration/)
#### [A06:2021-Vulnerable and Outdated Components](https://owasp.org/Top10/A06_2021-Vulnerable_and_Outdated_Components/)
#### [A07:2021-Identification and Authentication Failures](https://owasp.org/Top10/A07_2021-Identification_and_Authentication_Failures/)
#### [A08:2021-Software and Data Integrity Failures](https://owasp.org/Top10/A08_2021-Software_and_Data_Integrity_Failures/)
#### [A09:2021-Security Logging and Monitoring Failures](https://owasp.org/Top10/A09_2021-Security_Logging_and_Monitoring_Failures/)
#### [A10:2021-Server-Side Request Forgery](https://owasp.org/Top10/A10_2021-Server-Side_Request_Forgery_%28SSRF%29/)

### ZAP - 웹페이지 취약점 분석 툴
- [공식 홈페이지](https://www.zaproxy.org/)
- 오픈소스 웹 앱 스캐너 중 가장 활발하게 개발되고 있는 툴(https://github.com/psiinon/open-source-web-scanners)

### 과기부-KISA 취약점 점검 가이드 (2021. 03.)
- 제목: 주요 정보통신 기반시설 기술적 취약점 분석 평가 방법 상세 가이드
- 총 10대 분야: (1) Unix, (2) Windows, (3) 보안장비, (4) 네트워크 장비, (5) 제어시스템, (6) PC, (7) 데이터베이스, (8) Web, (9) 이동통신, (10) 클라우드(VMWare, KVM 등 가상화 장비)
#### (8) 웹 취약점 분석평가 항목
##### 버퍼 오버플로우(BO)
- 내용: 사용자 입력 값의 문자열 길이 제한
- 목적: 사용자 입력값의 길이가 비정상적인 경우 발생할 수 있는 오류 차단
- 위협: 길이 제한 없는 입력값에 의하여 저장공간을 초과하거나, 오류 발생에 따른 정보 노출 혹은 비인가 접근 등이 발생할 수 있음.
- 대상: 웹 애플리케이션 소스코드
- 점검 방법
    1) 로그인 페이지 계정 정보 입력란에 대량의 문자열 넣어보기
    2) 이용자 정보변경 페이지에서 여러 항목에 대량의 문자열 넣어보기
    3) 검색란에 대량의 문자열 넣어보기
    4) 게시글 본문에 대량의 문자열 넣어보기
    5) URL 파라미터 값에 대량의 문자열 넣어보기
- 조치 방안
    - 웹 서버, 웹 앱 등 서버 최신 버전으로 설치
    - 입력값 파라미터 길이 제한하기
    - (?) 동적 메모리 할당을 위해 크기를 사용하는 경우 그 값이 음수가 아닌지 검사하여 버퍼 오버플로우를 예방하는 형태로 소스 코드 변경
    - 버퍼 오버플로우를 탐지하는 스캐닝 툴로 주기적으로 점검


# 소프트웨어 공학
## [Software Development Life Cycle (SDLC)](https://www.geeksforgeeks.org/software-development-life-cycle-sdlc/?ref=lbp#article-meta-div)
소프트웨어 개발 생애주기 6단계
1. 계획 수립과 요구사항 분석
2. 요구사항 정의
3. 아키텍처 설계
4. 제품 개발
5. 테스트와 통합
6. 배포와 유지보수


## Engineering Minds
### Software Engineer KPI
[A chatGPT answer](https://chat.openai.com/share/57dba6aa-d4e1-402a-a483-4dd6518ccaca)
`Q`: KPI for software engineers
```
Key Performance Indicators (KPIs) are quantifiable measurements used to evaluate the performance of individuals or teams against specific goals and objectives. For software engineers, KPIs are used to track progress and success in their roles, which involve designing, developing, testing, and maintaining software applications. Here are some common KPIs for software engineers:

1.  Code Quality:   
    -   Code review feedback: Feedback from peers on code readability, maintainability, and adherence to best practices.
    -   Technical debt: Amount of work needed to fix code issues that could cause problems in the future.
    -   Code coverage: Percentage of code that is tested by automated tests.

2.  Productivity:
    -   Lines of code (LOC) written: Although a simple metric, it can help gauge the volume of work completed.
    -   Number of tasks completed: The total number of tasks, user stories, or issues resolved within a specific period.
    -   Lead time: The time it takes from task creation to completion.

3.  Timeliness:    
    -   On-time delivery: Percentage of tasks completed within the expected deadline or sprint.
    -   Time to resolve bugs: Average time it takes to fix bugs or defects.

4.  Collaboration and Communication:   
    -   Cross-functional collaboration: How well a software engineer works with team members from other disciplines.
    -   Contribution to team discussions: Participation in technical discussions, team meetings, and decision-making processes.

5.  Continuous Learning and Improvement:    
    -   Skill development: Regularly updating skills through learning, attending courses, or acquiring certifications.
    -   Implementation of new ideas and technologies: Proactively bringing new ideas, tools, or practices to the team to improve development processes.

6.  Customer and Stakeholder Satisfaction:   
    -   Internal stakeholder satisfaction: Satisfaction of project managers, team members, and other relevant stakeholders.
    -   External customer satisfaction: Feedback from end-users on the usability, functionality, and reliability of the software product.

It is important to note that KPIs should be customized to the context of each organization or project and aligned with the company's overall objectives. A balanced set of KPIs will help provide a comprehensive evaluation of a software engineer's performance.
```
`Q`: I think it would be a good idea to implement KPI for myself to manage my career.
```
It's an excellent idea to establish personal KPIs to manage your career, even if your company does not have a formal process in place. By setting measurable goals, you can track your progress and ensure continuous improvement. Here are some steps to help you create your KPIs:

1.  Align with company goals and objectives: Make sure your personal KPIs align with your company's goals, your department's objectives, and your role's requirements. This will ensure you're working towards the same overall vision and contributing to the success of the organization.

2.  Identify key areas of focus: Determine the critical areas you want to improve upon or maintain high performance in. These could be technical skills, soft skills, or specific processes. For example, as a software engineer, your key areas of focus may include code quality, productivity, learning and development, and collaboration.

3.  Set SMART goals: Create specific, measurable, achievable, relevant, and time-bound (SMART) goals for each key area of focus. This will make it easier to track your progress and adjust your goals as needed.
For example, some personal KPIs for a software engineer could include:
-   Code Quality: Achieve a minimum of 85% code coverage for all new projects within the next 6 months.
-   Productivity: Complete an average of 8 tasks per week over the next quarter.
-   Learning and Development: Obtain a new certification or complete a relevant course in your field within the next year.
-   Collaboration: Actively participate in at least one cross-functional project or initiative per quarter.

4.  Track your progress: Regularly monitor your performance against your personal KPIs. This can be done by maintaining a simple spreadsheet or using a project management tool. Evaluate your progress, identifying areas of improvement or where adjustments might be needed.

5.  Reflect and adjust: Periodically review your KPIs to ensure they still align with your personal and professional growth goals. Be prepared to adjust your goals or set new ones as your career evolves or as you achieve specific milestones.

6.  Seek feedback: If possible, share your KPIs with your manager or colleagues to gather their insights and feedback. This can help you validate your goals, identify blind spots, and receive additional support and encouragement.
    
By taking control of your personal KPIs, you'll be better positioned to manage your career and demonstrate your value to the organization, even without a formal performance management process in place.
```

[13 Software Development KPIs Every Dev Team Should Track](https://www.datapad.io/blog/software-development-kpis)
KPI #1. Cycle Time
KPI #2. Development Velocity
KPI #3. Change Failure Rate
KPI #4. Deployment Frequency
KPI #5. Pull Request [PR] Size
KPI #6. Defect Detection Ratio [DDR]
KPI #7. Code Coverage Percentage
KPI #8. Code Churn
KPI #9. Code Simplicity
KPI #10. Cumulative Flow
KPI #11. Bug Rates
KPI #12. Mean Time Between Failures [MTBF] and Mean Time to Repair [MTTR]
KPI #13. Net Promoter Score

## OOP
### 디자인 패턴
#### MVC 패턴
MVC 패턴은 웹 앱에 널리 쓰이는 디자인 패턴으로서 아래의 세 가지 요소로 구성된다.
-   Model: 데이터와 비즈니스 로직 관리
-   View: UI와 데이터 표현부
-   Controller: 이용자 입력의 처리와 Model과 View와의 상호작용

##### Model
필요한 데이터를 받아오거나, 웹 앱의 다른 부분에서 다루기 용이한 형태로 변환할 수 있다. `Node.js` 프로젝트에서는 주요 데이터베이스와의 CRUD, 외부 API 호출 등을 포함할 것이다.
##### View
이용자 부분에서 접하는 HTML, CSS, 그리고 JavaScript.
##### Controller
View로부터 요청을 처리하거나 Model로 전달한다. 또한 Model로부터 받은 데이터를 View와 이용자에게 전달한다.

##### [ChatGPT] Node.js `express` 프로젝트에 적용하기
일반적인 프로젝트 구조 예시
```
. 
├── .env 
├── .gitignore 
├── app.js  # express entry point
├── package.json 
├── README.md 
├── node_modules/ 
├── public/  # static files like HTML, CSS, JS
├── src/ 
│ ├── config/  # 설정 파일? database.js의 경우 클래스 정의, 접속 정보 불러오기, 연결 기능 담당
│ ├── controllers/  # View-Model 왔다리 갔다리
│ ├── middlewares/ 
│ ├── models/ 
│ ├── routes/  # route definitions of the API's endpoints
│ ├── services/  # 비즈니스 로직을 처리하기 위해 필요한 외부 서비스. Database, file system, external APIs
│ ├── utils/  # 기능성 모듈이나 파일
│ └── index.js 
├── tests/  # 테스트 스크립트
└── views/  # front-end 기능
```

#### [Behavioral] The chain of responsibility
- 참고: https://en.wikipedia.org/wiki/Chain-of-responsibility_pattern
- 참고: https://refactoring.guru/design-patterns/chain-of-responsibility
- (처리기 클래스로 이어진) **책임연쇄**, 혹은 **책임의 사슬**
    - 어느 객체가 요청을 처리기에 전달할 수 있으며, 책임의 사슬로 연결된 여러 처리기가 그 요청을 연쇄적으로 처리함
    - 요청을 받은 처리기는 그 요청을 처리할지, 다음 처리기로 넘길지 결정함
- 언제 사용할까?
    - 프로그램이 서로 다른 종류의 요청을 다양한 방법으로 처리하지만, 사전에 요청의 종류나 작업순서를 알 수 없을 때
    - 여러 처리기를 특정한 순서로 실행해야 할 때
    - 처리기의 집합과 그들의 순서가 런타임에서 변해야 할 때

```Python
# 추상적인 Base 클래스
class Handler:
    def __init__(self, next_handler=None):
        self.next_handler = next_handler

    def handle_request(self):
        raise NotImplementedError()

class PartialProcessHandler(Handler):
    def handle_request(self):
        # 처리 작업

        if self.next_handler is None:
            # 다음 처리기가 정의되지 않았다면?
            print("작업 완료")
        
        else:
            # 다음 처리기의 handle_request() 메소드 호출
            self.next_handler.handle_request()

# creating handler objects
first_handler = FirstHandler()
...
partial_handler = PartialProcessHandler()
...
final_handler = FinalHander()

# chaining the handlers
first_handler.next_handler = partial_handler
...
partial_handler.next_handler = final_hander

# executing
first_handler.handle_request()
```

## 데이터베이스
## MySQL
### Trigger
### Event Scheduler
- MySQL official manual [25.4 Using the Event Scheduler](https://dev.mysql.com/doc/refman/8.0/en/event-scheduler.html)
- schedule에 의해 실행되는 '예약/반복 명령' (Linux `at`, `cron`)
#### 이벤트 생성하기: `EVENT ... ON SCHEDULE ...`
- 사전에 `event scheduler`가 활성화되어 있어야 함 -> [(참고) 25.4.2 Event Scheduler Configuration](https://dev.mysql.com/doc/refman/8.0/en/events-configuration.html)
	- `SHOW PROCESSLIST;` 실행 시 `User: event_scheduler` 있는지?
	- 없으면 활성화하기 -> `SET  GLOBAL event_scheduler =  ON;`
- Full syntax
	```SQL
	CREATE  
		[DEFINER  =  _user_]  
		EVENT  
		[IF  NOT  EXISTS]  
		_event_name_  
		ON  SCHEDULE  _schedule_  
		[ON  COMPLETION  [NOT]  PRESERVE]  
		[ENABLE  |  DISABLE  |  DISABLE  ON  SLAVE]  
		[COMMENT  '_string_']  
		DO  _event_body_;  

	_schedule_: { 
		AT  _timestamp_  [+  INTERVAL  _interval_]  ...  
		|  EVERY  _interval_  
		[STARTS  _timestamp_  [+  INTERVAL  _interval_]  ...]  
		[ENDS  _timestamp_  [+  INTERVAL  _interval_]  ...] 
	} 

	_interval_: 
		_quantity_ {YEAR  |  QUARTER  |  MONTH  |  DAY  |  HOUR  |  MINUTE  |  
					WEEK  |  SECOND  |  YEAR_MONTH  |  DAY_HOUR  |  DAY_MINUTE  |  
					DAY_SECOND  |  HOUR_MINUTE  |  HOUR_SECOND  |  MINUTE_SECOND}
	```
#### 이벤트 목록보기: `SHOW EVENTS`
## 소프트웨어 테스트 기법
1. 테스트 대상의 역할이 무엇인지 정의
- 가능한 모든 실패의 경우, edge cases 고려하기
- 예시: 네트워크 공유 폴더에 파일 복사하는 기능
	-  원본 파일이 없다면? 
	- 네트워크 연결이 끊어졌다면? 
	- 도착지에 충분한 용량이 없다면? 
	- 도착지에 파일이 이미 존재한다면?
2. 테스트 작성
- 테스트 프레임워크 등을 활용해 함수의 기능을 예상되는 결과물을 확인하는 테스트 작성하기
- 테스트는 함수가 예상대로 작동하였을 때의 조건(assertion)을 포함해야 함
- 예시
	- 복사 여부 체크
	- 원본 파일 존재 여부 체크
	- 네트워크 연결 상태 체크
	- 도착지 여유 용량 vs 파일 용량 체크
	- 도착지에 이미 존재하는지 체크
3. 함수 작성
- 위의 테스트가 모두 쓰여지고 나면 함수 작성 (당연히 테스트를 통과할 수 있게 작성)
4. 테스트 실행
- 함수 작성 후 테스트 실행. 모든 테스트 케이스를 통과할 때까지 수정
5. 리팩터
- 모든 테스트를 마치고 함수의 구조, 가독성, 성능 등을 향상할 수 있게 리팩터하기.
- 리팩터 이후에 테스트 실행하여 기능에 문제가 없는지 확인

### Unit Test 단위 테스트
#### Mock object 모조품
- 테스트 환경에서 실행할 때 실제 시스템과 분리된 부분
- 목표: to define the behavior and expectations of a dependency in a test scenario
- 예를 들면 어느 데이터베이스나 네트워크에 작용하는 함수를 테스트할 때 실제 대상 대신에 가짜 객체로 대체


# 프로그래밍 언어
## Python

### subprocess module - 자녀 프로세스 실행 및 관리
> subprocess 모듈은 새로운 프로세스를 생성하고, 그들의 입력/출력/에러 파이프에 연결하고, 반환 코드를 얻을 수 있도록 합니다.
- `run()`: 자녀 프로세스 실행하고 종료될 때까지 대기. 처리할 수 있는 모든 사례에 `run()`으로 해결할 것.
- `Popen()`: `run()`의 하부(low-level) 인터페이스.

#### 기본적인 사용법
```python
import subprocess

subprocess.run(["ls", "-l"])

# 표준 출력(stdout)과 표준 에러(stderr) 캡처됨
subprocess.run(["ls", "-l", "/dev/null"], capture_output=True)  
```
- (참고) Python `subprocess`로 Python 앱을 실행시키는 것은 가능하긴 하지만 넌센스일 수도 있음. 하위 프로세스에서 실행할 특별한 이유가 있다면?
#### `CompletedProcess`: `run()` 종료 후 반환되는 인스턴스
    ```python
    import subprocess

    completed_process = subprocess.run(["ls", "-l"])
    completed_process.returncode
    # 0
    ```
- `check=True` 인자: 하위 프로세스가 실패할 경우 예외를 발생시킴. 기본값 `False`로서, 프로그램이 실패하면 반환값만 받아올 뿐 예외를 일으키진 않는다.
#### 예외
- `CalledProcessError`
- `TimeoutExpired`
- `FileNotFoundError`
```python
import subprocess

try:
    subprocess.run(
        ["python", "timer.py", "5"], timeout=10
    )
except FileNotFoundError as e:
    print(f"파일을 찾을 수 없습니다.\n{e}")
except subprocess.CalledProcessError as e:
    print(f"프로그램이 비정상적으로 종료되었습니다. 반환코드: {e.returncode}.\n{e}")
except subprocess.TimeoutExpired as e:
    print(f"프로세스 타임아웃.\n{e}")
```
### SQLAlchemy (ORM library)
> a comprehensive set of tools for working with databases and Python

ORM (Object Relational Mapper) + Core + SQL Expression Language
ORM은 말그대로 객체(object)와 관계(relation) 간의 매핑이다. 혹은 그러한 매핑을 구현하는 여러 기술과 도구를 포함한다. Java, Python 등의 객체지향 언어에서 어느 관계형 데이터베이스상의 개체(entity)를 대변하는 클래스를 작성할 수 있다. 클래스 메소드에는 CRUD에 대응하는 여러 데이터 조작 기능을 구현할 수 있다. 이러한 클래스로부터 호출된 객체는 데이터베이스에 존재하거나 생성될 실제 데이터 단위를 나타낸다. 프로그래밍 언어의 객체와 관계형 데이터베이스의 데이터가 서로 매핑된 것이다. `SQLAlchemy`와 같은 ORM 라이브러리는 매핑을 간편하게 구현할 수 있도록 도와주는 도구의 모음이다. 아래는 `User` 클래스와 `users` 테이블을 매핑하는 코드.
```Python
from sqlalchemy import create_engine 
from sqlalchemy.ext.declarative import declarative_base 
from sqlalchemy import Column, Integer, String 

engine = create_engine('sqlite:///example.db', echo=True) 
Base = declarative_base()

class  User(Base):
	__tablename__ = 'users'  

	id = Column(Integer, primary_key=True) 
	name = Column(String) 
	email = Column(String) 
	
	def  __repr__(self): 
		return  f"User(id={self.id}, name={self.name}, email={self.email})"
```

```Python
Base.metadata.create_all(engine)
```

```Python
from sqlalchemy.orm import Session 

# start session 
session = Session(engine) 

# create user 
new_user = User(name="Alice", email="alice@example.com") 
session.add(new_user) 

# commit the transaction 
session.commit() 
print(new_user.id) # prints the new user's id
```
```Python
# Query the database 
users = session.query(User).all() 

for user in users: 
	print(user)
```
### dotenv
- [pypi 홈페이지](https://pypi.org/project/python-dotenv/)
- `.env` 파일로부터 key-value 쌍을 읽어와서 환경변수로 설정하는 외부 모듈. 
- [12 Factor](https://12factor.net/) 방법론에서 [Ⅲ.Config](https://12factor.net/ko/config)의 준수를 돕는다. 
    - Ⅲ.Config: 앱 설정 값은 환경변수에 저장.
- 설치
    ```
    pip install python-dotenv
    ```

#### 사용법
- `.env` 파일: key=value 형태로 여러 환경변수 저장. 문자열로 불러옴.
    ```
    LOGGING=INFO
    ROOT_DIR="~/app"
    DB_HOST="10.0.0.0"
    ```
- `load_dotenv()`로 불러오면 프로세스 내에서 환경변수 값에 접근할 수 있음.
    ```Python
    from dotenv import load_dotenv
    load_dotenv()

    # os.environ 이나 os.getenv 등으로 불러오기
    import os
    env_value = os.environ.get("<.env key name>")
    ```
- `git` 프로젝트일 경우 `.gitignore`에 `.env` 추가하여 보안상 민감한 설정값을 원격 저장소에 올라가지 않게 하자.

## JavaScript
### Syntax
#### 화살표 함수
기존의 함수 표현식에 비해 간결함

- `this`, `arguments`, `super`에 대한 바인딩을 갖지 않으며, methods로 쓰일 수 없음
- constructor로 쓰일 수 없음 -> `new`에 던지면 `TypeError` 반환. `new.target` 키워드에 접근 권한 없음
- 표현식에 `yield`를 사용할 수 없고, 제너레이터 함수로서 만들어질 수도 없음.
```js
() => expression

param => expression

(param) => expression

(param1, paramN) => expression

() => {
  statements
}

param => {
  statements
}

(param1, paramN) => {
  statements
}
```
나머지 인수, 기본값 인수, destructuring 지원
```js
(a, b, ...r) => expression
(a = 400, b = 20, c) => expression
([a, b] = [10, 20]) => expression
({ a, b } = { a: 10, b: 20 }) => expression
```
`async`와 함께 쓸 때:
```js
async param => expression
async (param1, param2, ...paramN) => {
  statements
}
```

### [Array](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/Array)
원시형(primitive*)이 아님. 즉 `Array`라는 클래스로서 메소드나 속성을 갖는다.
*primitive: object가 아닌 자료형으로서, 메소드나 속성 또한 갖지 않는다. `string`, `number`, `bigint`, `boolean`, `undefined`, `symbol`, `null`

`Array` 객체의 핵심적인 특성
- resizable & type-mix of elements: 원소를 추가/제거할 수 있고 서로 다른 자료형의 원소를 담을 수도 있음. (<-> [typed arrays](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Guide/Typed_arrays))
- not associative: 음이 아닌 정수만을 index로 사용할 수 있음
- zero-indexed: 첫번째 원소의 index=0, 두번째는 1, ..., 맨 마지막 원소는 `Array.length - 1`
- 모든 배열-복사 기능은 `shallow copy`** 만듦

**shallow copy: 어느 객체의 `shallow copy`는 그 속성이 원본 객체의 속성과 동일한 참조를 공유한다. 즉 원본 객체의 속성값이 바라보는 곳과 복사본 객체의 속성값이 바라보는 곳이 동일하다. 이 때 둘 중 어느 하나의 속성값을 변경할 경우, 두 객체 모두 값이 달라진 것처럼 보인다. 이와 달리 `deep copy`는 원본과 독립적인 복사본 객체를 만들어낸다.

#### 인덱스: `배열명[]`
sparse arrays: 공란(empty slots)을 포함한 배열
길이는 n인데 실제 값이 있는 원소는 n개 미만인 경우

#### 여러 가지 `Array` 메소드
-   [`concat()`](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/Array/concat)
-   [`copyWithin()`](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/Array/copyWithin)
-   [`every()`](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/Array/every)
-   [`filter()`](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/Array/filter)
-   [`flat()`](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/Array/flat)
-   [`flatMap()`](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/Array/flatMap)
-   [`forEach()`](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/Array/forEach)
-   [`indexOf()`](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/Array/indexOf)
-   [`lastIndexOf()`](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/Array/lastIndexOf)
-   [`map()`](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/Array/map)
-   [`reduce()`](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/Array/reduce)
-   [`reduceRight()`](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/Array/reduceRight)
-   [`reverse()`](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/Array/reverse)
-   [`slice()`](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/Array/slice)
-   [`some()`](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/Array/some)
-   [`sort()`](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/Array/sort)
-   [`splice()`](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/Array/splice)
아래부터는 empty slot을  `undefined`로 처리하는 젊은 메소드.
-   [`entries()`](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/Array/entries)
-   [`fill()`](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/Array/fill)
-   [`find()`](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/Array/find)
-   [`findIndex()`](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/Array/findIndex)
-   [`findLast()`](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/Array/findLast)
-   [`findLastIndex()`](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/Array/findLastIndex)
-   [`group()`](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/Array/group)  Experimental
-   [`groupToMap()`](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/Array/groupToMap)  Experimental
-   [`includes()`](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/Array/includes)
-   [`join()`](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/Array/join)
-   [`keys()`](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/Array/keys)
-   [`toLocaleString()`](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/Array/toLocaleString)
-   [`values()`](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/Array/values)
#### filter()
주어진 배열의 `shallow copy`를 만들고, 조건을 만족하는 요소만 추려서 새로운 배열을 반환한다. 새로운 배열을 반환하기 때문에, `shallow copy`된 원본 배열은 영향을 받지 않는다.
```js
const words = ['spray', 'limit', 'elite', 'exuberant', 'destruction', 'present'];

const result = words.filter(word => word.length > 6);

console.log(result);
// Expected output: Array ["exuberant", "destruction", "present"]
```
하지만 배열의 요소가 객체일 경우 `shallow copy`로 인한 원본 의존성이 문제가 될 수도 있다. `객체의 shallow copy`는 각각의 속성들이 원본 객체의 해당 속성과 동일한 참조를 공유한다. ([mdn 설명](https://developer.mozilla.org/en-US/docs/Glossary/Shallow_copy))
>a copy whose properties share the same [references](https://developer.mozilla.org/en-US/docs/Glossary/Object_reference) (point to the same underlying values) as those of the source object from which the copy was made.

그 결과 원본이나 복사본 중 어느 한 가지를 수정할 경우, 다른 한쪽도 마찬가지로 수정된 값을 참조할 것이다. 
이와 반대로 `deep copy`는 원본과 복사본이 서로 독립적임. 객체로 이루어진 어느 배열에 대해서 `deep copy` 형식의 `filter()` 기능을 수행하려면, 각각의 객체를 `JSON.parse(JSON.stringify())`와 같은 방법으로 `deep copy`하고, 새로운 배열에 담는다. 아래는 `Array` 클래스에 `deepFilter`라는 메소드를 정의하는 예문.
```js
Array.prototype.deepFilter = function(predicate) { 
	return this.reduce((acc, val) => { 
		if ( predicate(val) ) { 	
			acc.push(JSON.parse(JSON.stringify(val)));
			} 
		return acc; }, []);
};
```

### NodeJS
#### winston- a logger for just about everyhthing
- [npm 페이지(https://www.npmjs.com/package/winston)](https://www.npmjs.com/package/winston)
- npm 간판 로깅 패키지
```JavaScript
// logger.js 파일에 logger class 정의, 설정 등 몰아넣으면 편리함
// 다른 모듈에서 require('./logger') 불러오기
const winston = require('winston');
const { combine, errors, timestamp, printf, colorize, align, simple } = winston.format;

const logger = winston.createLogger({
  // 'debug' 수준 이상의 로그를 모두 기록합니다.
  level: 'debug',

  // winston.format.combine() => 여러 format 형식 조합
  // winston.format 여러 format 설정할 수 있는 모듈, 메소드
  format: combine(
    errors( { stack: true }),
    timestamp({format: 'YYYY-MM-DD hh:mm:ss.SSS A'}),
    align(),
    printf((info) => `[${info.timestamp}] ${info.level}: ${info.message}`)
    ),
  transports: [
    // level별로 여러 파일에 나누어 기록할 수 있음
    new winston.transports.File({ filename: './logs/error.log', level: 'error' }),
    new winston.transports.File({ filename: './logs/combined.log' }),
  ],
  
  // 아래는 미처리 예외나 promise 거절 로그 기록하는 처리기
  exceptionHandlers: [
    new winston.transports.File( { filename: './logs/exception.log' }),
  ],
  rejectionHandlers: [
    new winston.transports.File( { filename: './logs/rejections.log' })
  ]
});

// production 아닐 경우, console 출력용 logger 추가함
if (process.env.NODE_ENV !== 'production') {
  logger.add(new winston.transports.Console({
    format: winston.format.combine(
      colorize(),
      simple()
    ),
  }));
}

// 여러 모듈에서 불러올 수 있게 내보내기.
module.exports = logger;
```

```javascript
// 예시: main.js 등에서:
const logger = require('./logger');
logger.info('실행 완료')
```

##### 기본 logging 레벨
- 괄호 안은 우선순위
    (0) error: 치명적, 예상하지 못한 오류.
    (1) worn: 경고. error날 가능성.
    (2) info: 일반적인 정보 메시지. high-level 상세 정보.
    (3) verbose: info보다는 더 상세하고 잡다한 정보.
    (4) debug: 디버깅이나 trouble-shooting에 필요한 정보. 프로그램 내부적인 상태나 변수 값에 대한 정보.
    (5) silly: 너무 사소하거나 중요하지 않은 정보들.

### jest - JavaScript Testing Framework
- [공식 웹사이트](https://jestjs.io/)

#### 설치 및 간단 사용법
1. 프로젝트 경로에서 설치
```
npm install --save-dev jest
```
2. 테스트 모듈 생성
- 테스트 대상: `sum.js`
```js
function sum(a, b) {
  return a + b;
}
module.exports = sum;
```
- 테스트 모듈: `sum.test.js`
```js
// 테스트 대상 불러오기
const sum = require('./sum');

// 테스트 케이스 작성: expect, toBe
test('adds 1 + 2 to equal 3', () => {
  expect(sum(1, 2)).toBe(3);
});
```
3. `package.json` 수정
```json
{
  "scripts": {
    "test": "jest"
  }
}
```
4. 실행
```
npm test
...
PASS  ./sum.test.js
✓ adds 1 + 2 to equal 3 (5ms)
```

#### matchers - `expect` 클래스의 여러 판단 메소드
- `toBe` (`Object.is`): exact equality
- `toEqual`: value comparision. object 비교 시 다음에 해당하는 key는 무시함
    - `undefined` properties
    - `undefined` array items
    - array sparseness
    - object type mismatch
- `toStrictEqual`: object를 비교할 때 `toEqual`과 달리, 모든 key에 대해 strict comparision
- `not`: 용례  
```js
test('adding positive numbers is not zero', () => {
  for (let a = 1; a < 10; a++) {
    for (let b = 1; b < 10; b++) {
      expect(a + b).not.toBe(0);
    }
  }
});
```
- `toBeNull`, `toBeUndefined`, `toBeDefined`: 각각 `null`, `undefined`, `define` 판별
- 조건식 판별: `toBeTruthy`, `toBeFalsy`: 각각 `true`, `false` 
- 대소 비교: `toBeGreaterThan`, `toBeGreaterThanOrEqual`, `toBeLessThan`, `toBeLessThanOrEqual`
- 문자열, 정규식; `toMatch`
- 반복가능자: `toContain`
- 예외, 오류 발생 여부: `toThrow`
- 그밖의 `expect` 요소: [API docs](https://jestjs.io/docs/expect)
### JSDoc - a markup language used for annotating JavaScript code files.
- [공식 홈페이지](https://jsdoc.app/)
- JavaScript 코드에서 입출력 타입 정보를 관리하는 문서화 기능
#### 백문불여일견

```js
/** 아래 함수에 대한 설명 */
function foo() {

}

/** 
 * 한 권의 책
 * @constructor
*/
function Book(title, author) {
}

/** 
 * 한 권의 책
 * @constructor
 * @param {string} title - 제목
 * @param {string} author - 저자
*/
function Book(title, author) {
}
```
- `/** */` 사이에 텍스트와 태그(`@태그명`)를 곁들여 코드에 대한 여러 가지 부가정보를 제공
- `@param`, `@returns` 등 입출력 변수에 `{type}` 명시

위와 같이 설명을 첨가한 파일에 대해 아래의 명령어를 실행하면, `out/` 경로에 HTML 페이지를 자동으로 생성
```
jsdoc book.js
```

# AWS Cloud Computing
## AWS Batch
- AWS 클라우드 배치 작업 관리 서비스
    - 클라우드 기반 배치 작업: 동시에 여러 컴퓨팅 자원 활용

### AWS Batch의 주요 요소
1. 작업<sub>Jobs</sub>: 작업의 기본 단위(쉘 스크립트, 리눅스 실행파일, 도커 컨테이너 이미지 등). 
2. 작업 정의<sub>Job Definitions</sub>: 작업 실행 방법. IAM을 통해 AWS 자원에 대한 접근 제공.
3. 작업 대기열<sub>Job Queues</sub>: 작업 요청 목록 관리.
4. 컴퓨팅 환경<sub>Compute Environment</sub>: 작업을 실행하는 컴퓨팅 자원. 특정 EC2 인스턴스 유형 지정할 수 있음. 일반적인 EC2 자원도 사용 가능하며, AWS Fargate(컨테이너 오케스트레이션)도 사용 가능.

### 사전 준비
- AWS 계정 만들기
- 관리자 사용자
- IAM 역할 만들기 -> 컴퓨팅 환경과 컨테이너 자원에 대한 접근권한
- 키쌍 만들기
- VPC 만들기 Virtual Private Cloud
- 보안그룹 만들기
- AWS CLI 설치하기

### 시작하기
#### 컴퓨팅 환경 만들기
- 컴퓨팅 환경 3종: Fargate, EC2, EKS
    - Fargate: 단위 작업마다 가벼운 인스턴스를 새로 시작함. 부팅 시간은 EC2보다 짧을 수 있겠지만, 작업 개수가 많아지면 불리할 수도 있음.
    - EC2: 특정 프로세서, 인스턴스 유형, 커스텀 AMI(Amazon Machine Image)가 필요하거나 대규모 워크로드 다룰 때. 이미 작동 중인 EC2에 작업 전달. AMI는 `Amazon Linux 2` 기반 사용 가능
    - EKS

- 오케스트레이션 유형
    - 관리형: 컴퓨팅 리소스와 인스턴스 유형을 AWS가 관리함. EC2 온디맨드 인스턴스 혹은 EC2 스팟 인스턴스 중 한 가지 선택.
    - 비관리형: 컴퓨팅 환경을 직접 준비하고 제어함.

#### 작업 대기열 만들기
#### 작업 정의 만들기
#### 작업 만들기

# Linux Systems
## crontab
```
m h     dom mon dow      commands
```

# 컴퓨터 일반
## ssh 공개키 인증
- OpenSSH 서버 <-> 클라이언트 간 여러 가지 인증 방식 있음
- 공개키 인증 방식을 설정해두면 접속할 때 비밀번호를 입력하지 않아도 됨

1. SSH key 쌍 준비
    - 클라이언트에 SSH key 쌍을 만든다. (이미 있다면 pass)
        ```bash
        ssh-keygen -t rsa
        ```
    - `~/.ssh` 경로에 `id_rsa`, `id_rsa.pub` 파일 생김

2. 원격 서버에 공개키 저장
    - 원격 서버의 `~/.ssh/authorized_keys`에 공개키 값(`id_rsa.pub`)을 복사한다.
        ```bash
        ssh-copy-id username@remote-server
        ```
    - 위의 명령어를 실행하거나, 공개키 값을 직접 복사/붙여넣기 해도 됨
    - 키 파일을 옮길 때에는 암호화를 지원하는 안전한 방식으로 복사할 것.

3. ssh 연결 확인
    - 클라이언트에서 원격 서버로 ssh 접속을 시도한다.
        ```bash
        ssh username@remote-server
        ```
    - 여전히 비밀번호를 요구한다면 아래의 파일 권한을 확인해본다.

4. authorized_keys 상위 경로 권한 확인
    - `~`: 755 (사용자 home 경로)
    - `~/.ssh`: 700
    - `~/.ssh/authorized_keys`: 600
    - (참고) r(읽기) - 4, w(쓰기) - 2, x(실행) - 1
    
## Git
### git add
- staging 영역에 파일을 추가함.
- `git add .`: 작업경로에 있는 파일 중 새로 추가되거나 수정된 파일을 모두 staging함.
### git commit
- staged된 내용을 저장하고 새로운 스냅샷 만듦.
- `-a`: 수정된 모든 파일을 staging 영역에 넣은 뒤 commit 실행 (`git add . && git commit`)
- `-m \<commit 메시지>`
- `git commit -am "commit 메시지"`: 앞서 `git add` 하지 않았지만 변경된 내용들을 모두 commit함.

### git log
- 최근 commit 이력
- `--graph`: Tree 그래프와 함께 표시
- `--format`: 로그 포맷 사용자화
- `--all`: refs, tags, branches 모두 표시

### git show
- `git show <commit id>`: 해당 커밋의 변경 내용 자세히 보여주기
- `--stat`: 변경 내용 요약한 수치
<!--stackedit_data:
eyJoaXN0b3J5IjpbMTA1NTYyMTI0NCwtMTExMzc0MjIyM119
-->