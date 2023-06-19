# TIL
진정한 자유를 위한 기록
평일 17:00, 주말 random
commit 메시지 포멧: yyyy-mm-dd TIL 루틴


# 여러 가지 생산성 툴
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
# 정보보호/보안
## 취약점 진단/분석
- 법적 근거: `「개인정보의 안전성 확보조치 기준」 제6조(접근통제)` ④ 고유식별정보를 처리하는 개인정보처리자는 인터넷 홈페이지를 통해 고유식별정보가 유출ㆍ변조ㆍ훼손되지 않도록 연 1회 이상 취약점을 점검하고 필요한 보완 조치를 하여야 한다.

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
## OOP
### 디자인 패턴
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

# 소프트웨어 공학
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
eyJoaXN0b3J5IjpbLTExMTM3NDIyMjNdfQ==
-->