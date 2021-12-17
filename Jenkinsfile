pipeline {
    agent {
        docker {
            image 'nokal/simod-testing:v1.1.0'
            args '-u root'
        }
    }
    stages {
        stage('Test') {
            steps {
                sh 'cd /usr/src && bash test_simod.sh'
            }
        }
    }
}
