-- phpMyAdmin SQL Dump
-- version 5.0.2
-- https://www.phpmyadmin.net/
--
-- Host: 127.0.0.1
-- Generation Time: Dec 22, 2020 at 03:03 PM
-- Server version: 10.4.14-MariaDB
-- PHP Version: 7.2.33

SET SQL_MODE = "NO_AUTO_VALUE_ON_ZERO";
START TRANSACTION;
SET time_zone = "+00:00";


/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET @OLD_CHARACTER_SET_RESULTS=@@CHARACTER_SET_RESULTS */;
/*!40101 SET @OLD_COLLATION_CONNECTION=@@COLLATION_CONNECTION */;
/*!40101 SET NAMES utf8mb4 */;

--
-- Database: `gtbit`
--

-- --------------------------------------------------------

--
-- Table structure for table `contacts`
--

CREATE TABLE `contacts` (
  `sno` int(11) NOT NULL,
  `name` text NOT NULL,
  `phone_num` varchar(50) NOT NULL,
  `msg` text NOT NULL,
  `date` datetime DEFAULT current_timestamp(),
  `email` varchar(50) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

--
-- Dumping data for table `contacts`
--

INSERT INTO `contacts` (`sno`, `name`, `phone_num`, `msg`, `date`, `email`) VALUES
(1, 'hello', '1234568979', 'first post', '2020-12-06 00:10:29', 'first@gmail.com'),
(2, 'helo', '7891231083', 'klklkl', '2020-12-06 00:30:16', 'firsr@gmail.com'),
(3, 'helo', '7891231083', 'klklkl', '2020-12-06 00:31:40', 'firsr@gmail.com'),
(4, 'Manthn', '9057288057', 'helloooff', '2020-12-06 00:38:14', 'gupta.manthan264@gmail.com'),
(5, 'Manthn', '09057288057', 'sfdfdfdf', '2020-12-06 00:38:28', 'gupta.manthan264@gmail.com'),
(6, 'fdfdf', '5', '', '2020-12-06 01:15:14', 'fdffd@gmail.com'),
(7, 'fdfdf', '7891231083', 'jhvidfhg', '2020-12-06 01:15:43', 'fdffd@gmail.com'),
(8, 'fdfdf', '7891231083', 'jhvidfhg', '2020-12-06 01:19:29', 'fdffd@gmail.com'),
(9, 'Manthn', '09057288057', 'fhthggdrf', '2020-12-09 19:40:06', 'gupta.manthan264@gmail.com'),
(10, 'manthan', '1212121212', 'hey friends', '2020-12-21 23:34:04', 'mgmanrhan264@gmail.com');

-- --------------------------------------------------------

--
-- Table structure for table `posts`
--

CREATE TABLE `posts` (
  `sno` int(11) NOT NULL,
  `title` text NOT NULL,
  `tagline` text NOT NULL,
  `slug` varchar(25) NOT NULL,
  `description` text NOT NULL,
  `content` text NOT NULL,
  `location` text NOT NULL,
  `profile` text NOT NULL,
  `salary` varchar(11) NOT NULL,
  `roles` text NOT NULL,
  `skills` text NOT NULL,
  `img_file` varchar(12) NOT NULL,
  `date` datetime NOT NULL DEFAULT current_timestamp(),
  `button` text DEFAULT NULL,
  `referral` text DEFAULT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

--
-- Dumping data for table `posts`
--

INSERT INTO `posts` (`sno`, `title`, `tagline`, `slug`, `description`, `content`, `location`, `profile`, `salary`, `roles`, `skills`, `img_file`, `date`, `button`, `referral`) VALUES
(1, 'Slack Off Campus Drive 2020', 'Off Campus Drive ', 'first-post', 'Slack is hiring for the position of Software Development Engineer. For insights and updates related to off-campus drives and internship drives be active on our website.', 'Slack is hiring for the position of Software Engineer, Frontend. For insights and updates related to off-campus drives and internship drives be active on our website.\r\nSlack is where work happens. It connects you with the people and apps you work with every day, no matter where you are or what you do. We believe everyone deserves to work in a welcoming, respectful, and empathetic culture. We live by our values and hire accordingly.', 'Bangalore, India\r\n', 'Software Development Engineer\r\n', '20 LPA', 'Works in requirements like Engineering, Design, Development, and Deployment\r\n,provides technical expertise in Big Data space (Apache Spark, Kafka, Flink, Hadoop Stack like M/R, HDFS, Hive, HBase, Flume, Sqoop, etc. NoSQL stores like Cassandra, HBase etc), provides strong technical leadership in adopting and contributing to open source technologies related to Big Data. ', 'Experience in programming languages (Java, Ruby, JavaScript, HTML5, CSS)\r\nExperience working with databases preferably Postgres,Cassandra.                                                         \r\nGood knowledge working with REST API, Microservices.\r\nExperience with frameworks (Spring Boot, Ruby on Rails).\r\nGood understand of DevOps tools (Docker, Kubernetes, Ansible, AWS).\r\nExperience working in Agile/Scrum environment.', 'slack.jpg', '2020-12-22 18:19:28', 'https://slack.com/intl/en-in/careers/2338092/software-engineer-frontend?gh_src=pmw542v31', 'https://www.linkedin.com/search/results/people/?keywords=slack&origin=SUGGESTION'),
(2, 'Amazon is hiring!', 'On Campus', 'second-post', 'Are you excited by the challenge of launching new business initiatives for Amazon in India which is one of the fastest growing Marketplace ? \r\nAmazon is hiring SDE-1', 'As a Software Development Engineer working in India Accounting Engineering Team, you will lead designing and building platforms that is configuration driven, self-service capable, to serve the fast growing needs of various businesses at Amazon. These systems play a critical role in enabling the business to launch on time, and handling real time business events and translate them into financial events that could then be persisted onto Financial Systems within Amazon.\r\nAs an engineer here, you will be part of a team of high performing engineers to build, scale, monitor, and optimize the platforms that processes these business and financial events. You will interface with several teams including Fulfillment Services, Fees, Taxes, Retail Systems, Marketplace Systems, Invoicing Platform, Digital Systems, Amazon Web Services, Financial Systems, Accounting and Operations.', 'Bangalore, India', 'Software Development Engineer 1', '30.25 LPA', 'Rich hands-on programming experience in Java or another object oriented language\r\n- Proven track record building large-scale systems and solving complex problems\r\n- Experience mentoring and developing the technical skills of Software Engineers\r\n- Deep understanding of design-patterns and putting them to use in real-world scenarios\r\n- Experience with service oriented technology\r\n- Experience working with and building scalable systems\r\n- Knowledge of AWS or other cloud platform\r\n- Experience developing in a Linux environment\r\n- Clear, professional communication skills able to contribute to team discussions, knowing when to contribute, when to listen, and when to ask questions.', 'Experience with one or more JavaScript Frameworks such as React or Angluar.\r\n· Ability to convert a design document into a working user interface\r\n· BS/MS in Computer Science or equivalent\r\n· Demonstrated ability to achieve stretch goals in a highly innovative and fast paced environment\r\n· Expertise in delivering high-quality, innovative application\r\n· Strong desire to build, sense of ownership, urgency, and drive\r\n· Excellent written communication and verbal agility are strong assets\r\n· - Prior experience with building and supporting live high-volume transaction processing systems\r\n- Familiarity with RDBMS\r\n- XML, ion and Web Service concepts\r\n- Agile development methodologies\r\n- Familiarity with UNIX or Linux systems', 'amazon.jpg', '2020-12-22 18:37:08', 'https://www.amazon.jobs/en/jobs/1303233/sde-i', 'https://www.linkedin.com/search/results/people/?keywords=amazon&origin=SWITCH_SEARCH_VERTICAL'),
(3, 'ZS On Campus drive 2020 ', 'On Campus ', 'third-post', 'ZS is hiring for the position of Decision Analytics Associate. For insights and updates related to off-campus drives and internship drives be active on our website.\r\n\r\nZS is a professional services firm that works side by side with companies to help develop and deliver products that drive customer value and company results. From R&D to portfolio strategy, customer insights, marketing and sales strategy, operations and technology, we leverage our deep industry expertise and leading-edge analytics to create solutions that work in the real world. Our most valuable asset is our people—a fact that’s reflected in our values-driven organization in which new perspectives are integral and new ideas are celebrated. ZSers are passionately committed to helping companies and their customers thrive in industries ranging from healthcare and life sciences, to high-tech, financial services, travel and transportation, and beyond.', 'ZS’s India Capability & Expertise Center (CEC) houses more than 60% of ZS people across three offices in New Delhi, Pune and Bengaluru. Our teams work with colleagues across North America, Europe and East Asia to create and deliver real world solutions to the clients who drive our business. The CEC maintains standards of analytical, operational and technological excellence across our capability groups. Together, our collective knowledge enables each ZS team to deliver superior results to our clients.\r\n\r\nZS’s Business Consulting group partners with clients to design and deliver solutions to help them tackle a broad range of business challenges. Our teams work on multiple projects simultaneously, leveraging advanced data analytics and problem-solving techniques. Our recommendations and solutions are based on rigorous research and analysis underpinned by deep expertise and thought leadership.', 'Pune, India', 'Business Analyst', '8-10 LPA', 'Develop advanced statistical models that relate marketing or sales activities to financial results\r\nSimulate the impact of a client’s future marketing or sales activities on their future real-world results\r\nLeverage optimization techniques to guide clients and ZS teams to the best decisions\r\nDesign custom analysis tools in applications such as Visual Basic, MS Excel, MS Access, SAS and ZS proprietary software\r\nSynthesize and communicate results to clients and ZS teams\r\nCollaborate with client and ZS teams to implement solutions\r\nGain immediate responsibility for project deliverables', 'Strong work ethic and personal initiative\r\nStructured problem solving skills\r\nFocus on quality and detail orientation\r\nExcellent organizational and task management skills\r\nStrong communication and business acumen\r\nAbility to work across global cross-office teams\r\nClient service orientation', 'zs.jpg', '2020-12-14 22:08:31', 'https://jobs.zs.com/India/job/Pune-17DAA-Pune-MH/411255900', 'https://www.linkedin.com/search/results/people/?keywords=zs%20associates&origin=SUGGESTION'),
(4, 'Synopsys Off Campus drive 2020', 'Off Campus ', 'fourth-post', 'Synopsys is hiring for the position of R&D Engineer. For insights and updates related to off-campus drives and internship drives be active on our website.', 'At Synopsys, we’re at the heart of the innovations that change the way we work and play. Self-driving cars. Artificial Intelligence. The cloud. 5G. The Internet of Things. These breakthroughs are ushering in the Era of Smart Everything. And we’re powering it all with the world’s most advanced technologies for chip design and software security. If you share our passion for innovation, we want to meet you.\r\n\r\nOur Silicon Design & Verification business is all about building high-performance silicon chips—faster. We’re the world’s leading provider of solutions for designing and verifying advanced silicon chips. And we design the next-generation processes and models needed to manufacture those chips. We enable our customers to optimize chips for power, cost, and performance—eliminating months off their project schedules.', 'Bangalore, India', 'R&D Engineer', '12-15 LPA', 'We’re looking for a R&D engineer in the ProtoCompiler R&D team in Bangalore for the following role.\r\nYou would be responsible for designing, developing, troubleshooting, debugging and maintaining large and efficient software systems for partitioning, logic, timing optimization, technology mapping steps of the FPGA prototyping software.', 'Given a requirement or functional specification, design and implement efficient data structures and algorithms in C/C++.\r\nWork with AE team in test planning, execution and customer support.\r\nB.Tech/M. Tech in CS/EE from a reputed institute.\r\nSound knowledge in data structures, graph algorithms and C/C++ programming on Windows/Unix.\r\nFamiliarity in digital logic design.', 'synopsys.jpg', '2020-12-22 18:58:02', 'https://sjobs.brassring.com/TGnewUI/Search/home/HomeWithPreLoad?PageType=JobDetails&partnerid=25235&siteid=5359&jobid=1545674#jobDetails=1545674_5359', 'https://www.linkedin.com/search/results/people/?keywords=synopsys&origin=SUGGESTION'),
(6, 'Google Off Campus drive 2020', 'Off Campus', 'fifth-post', 'Google is hiring for the position of Software Engineer. For insights and updates related to off-campus drives and internship drives be active on our website.', 'Google’s software engineers develop the next-generation technologies that change how billions of users connect, explore, and interact with information and one another. Our products need to handle information at massive scale, and extend well beyond web search.\r\nWe’re looking for engineers who bring fresh ideas from all areas, including information retrieval, distributed computing, large-scale system design, networking and data storage, security, artificial intelligence, natural language processing, UI design and mobile; the list goes on and is growing every day. As a software engineer, you will work on a specific project critical to Google’s needs with opportunities to switch teams and projects as you and our fast-paced business grow and evolve.', 'Hyderabad,Bengaluru', 'Software Engineer', '35-38 LPA', 'Research, conceive and develop software applications to extend and improve on Google’s product offering.\r\nContribute to a wide variety of projects utilizing natural language processing, artificial intelligence, data compression, machine learning and search technologies.\r\nCollaborate on scalability issues involving access to massive amounts of data and information.\r\nSolve challenges/problems that you are presented with.', 'Bachelor’s degree or equivalent practical experience.\r\nExperience working with Unix/Linux, Windows or Mac environments, distributed systems, machine learning, information retrieval and TCP/IP.\r\nExperience programming in C, C++, Java or Python.', 'google.jpg', '2020-12-22 19:13:42', 'https://careers.google.com/jobs/results/104192665471328966-software-engineer-university-graduate-2021-start/', 'https://www.linkedin.com/search/results/people/?keywords=google&origin=SUGGESTION');

--
-- Indexes for dumped tables
--

--
-- Indexes for table `contacts`
--
ALTER TABLE `contacts`
  ADD PRIMARY KEY (`sno`);

--
-- Indexes for table `posts`
--
ALTER TABLE `posts`
  ADD PRIMARY KEY (`sno`);

--
-- AUTO_INCREMENT for dumped tables
--

--
-- AUTO_INCREMENT for table `contacts`
--
ALTER TABLE `contacts`
  MODIFY `sno` int(11) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=11;

--
-- AUTO_INCREMENT for table `posts`
--
ALTER TABLE `posts`
  MODIFY `sno` int(11) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=7;
COMMIT;

/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
