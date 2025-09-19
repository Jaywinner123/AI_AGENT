# Clipper Agent Deployment Guide

This guide covers different deployment options for the Clipper Agent system, from local development to enterprise-scale production deployments.

## 🏠 Local Development

### Quick Start
```bash
# Clone repository
git clone https://github.com/your-org/clipper-agent.git
cd clipper-agent

# Install dependencies
pip install -r requirements.txt

# Create test video and run demo
python create_test_video.py
python clipper_demo_simple.py test_video.mp4
```

### Development Environment Setup
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows

# Install development dependencies
pip install -r requirements.txt

# Install pre-commit hooks (optional)
pip install pre-commit
pre-commit install

# Run tests
python -m pytest tests/ -v
```

### System Requirements
- **CPU**: 2+ cores (4+ recommended)
- **RAM**: 4GB minimum (8GB+ recommended)
- **Storage**: 10GB+ free space
- **OS**: Linux, macOS, or Windows
- **Python**: 3.8+
- **FFmpeg**: Latest version

## 🖥️ Single Server Deployment

### VPS/Dedicated Server Setup

#### 1. Server Preparation
```bash
# Update system (Ubuntu/Debian)
sudo apt update && sudo apt upgrade -y

# Install system dependencies
sudo apt install -y python3 python3-pip python3-venv ffmpeg git

# Create application user
sudo useradd -m -s /bin/bash clipper
sudo usermod -aG sudo clipper
```

#### 2. Application Setup
```bash
# Switch to application user
sudo su - clipper

# Clone repository
git clone https://github.com/your-org/clipper-agent.git
cd clipper-agent

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Create configuration
cp config/config.yaml.example config/config.yaml
# Edit configuration as needed
```

#### 3. Service Configuration
```bash
# Create systemd service file
sudo tee /etc/systemd/system/clipper-agent.service << EOF
[Unit]
Description=Clipper Agent Service
After=network.target

[Service]
Type=simple
User=clipper
WorkingDirectory=/home/clipper/clipper-agent
Environment=PATH=/home/clipper/clipper-agent/venv/bin
ExecStart=/home/clipper/clipper-agent/venv/bin/python clipper_demo_simple.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

# Enable and start service
sudo systemctl daemon-reload
sudo systemctl enable clipper-agent
sudo systemctl start clipper-agent
```

#### 4. Nginx Reverse Proxy (Optional)
```bash
# Install Nginx
sudo apt install -y nginx

# Create Nginx configuration
sudo tee /etc/nginx/sites-available/clipper-agent << EOF
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://localhost:8000;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
    }
}
EOF

# Enable site
sudo ln -s /etc/nginx/sites-available/clipper-agent /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl restart nginx
```

## 🐳 Docker Deployment

### Basic Docker Setup

#### 1. Create Dockerfile
```dockerfile
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create directories
RUN mkdir -p storage temp models

# Expose port
EXPOSE 8000

# Run application
CMD ["python", "clipper_demo_simple.py"]
```

#### 2. Build and Run
```bash
# Build image
docker build -t clipper-agent .

# Run container
docker run -d \
  --name clipper-agent \
  -p 8000:8000 \
  -v $(pwd)/storage:/app/storage \
  -v $(pwd)/config:/app/config \
  clipper-agent
```

### Docker Compose Setup

#### docker-compose.yml
```yaml
version: '3.8'

services:
  clipper-agent:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./storage:/app/storage
      - ./config:/app/config
      - ./temp:/app/temp
    environment:
      - CLIPPER_LOG_LEVEL=INFO
      - CLIPPER_STORAGE_PATH=/app/storage
    restart: unless-stopped
    depends_on:
      - redis
      - postgres

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    restart: unless-stopped

  postgres:
    image: postgres:15-alpine
    environment:
      - POSTGRES_DB=clipper_agent
      - POSTGRES_USER=clipper
      - POSTGRES_PASSWORD=your_password_here
    volumes:
      - postgres_data:/var/lib/postgresql/data
    ports:
      - "5432:5432"
    restart: unless-stopped

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/nginx/ssl
    depends_on:
      - clipper-agent
    restart: unless-stopped

volumes:
  redis_data:
  postgres_data:
```

#### Run with Docker Compose
```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f clipper-agent

# Stop services
docker-compose down
```

## ☁️ Cloud Deployment

### Google Cloud Platform

#### 1. Cloud Run Deployment
```bash
# Install Google Cloud SDK
curl https://sdk.cloud.google.com | bash
exec -l $SHELL
gcloud init

# Build and deploy
gcloud builds submit --tag gcr.io/PROJECT_ID/clipper-agent
gcloud run deploy clipper-agent \
  --image gcr.io/PROJECT_ID/clipper-agent \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --memory 2Gi \
  --cpu 2 \
  --timeout 3600
```

#### 2. GKE Deployment
```bash
# Create cluster
gcloud container clusters create clipper-cluster \
  --num-nodes=3 \
  --machine-type=n1-standard-2 \
  --zone=us-central1-a

# Get credentials
gcloud container clusters get-credentials clipper-cluster --zone=us-central1-a

# Deploy application
kubectl apply -f k8s/
```

### Amazon Web Services

#### 1. ECS Deployment
```bash
# Install AWS CLI
pip install awscli
aws configure

# Create ECR repository
aws ecr create-repository --repository-name clipper-agent

# Build and push image
$(aws ecr get-login --no-include-email --region us-east-1)
docker build -t clipper-agent .
docker tag clipper-agent:latest 123456789012.dkr.ecr.us-east-1.amazonaws.com/clipper-agent:latest
docker push 123456789012.dkr.ecr.us-east-1.amazonaws.com/clipper-agent:latest

# Create ECS service
aws ecs create-service \
  --cluster default \
  --service-name clipper-agent \
  --task-definition clipper-agent:1 \
  --desired-count 2
```

#### 2. EKS Deployment
```bash
# Install eksctl
curl --silent --location "https://github.com/weaveworks/eksctl/releases/latest/download/eksctl_$(uname -s)_amd64.tar.gz" | tar xz -C /tmp
sudo mv /tmp/eksctl /usr/local/bin

# Create cluster
eksctl create cluster --name clipper-cluster --region us-east-1 --nodes 3

# Deploy application
kubectl apply -f k8s/
```

### Microsoft Azure

#### 1. Container Instances
```bash
# Install Azure CLI
curl -sL https://aka.ms/InstallAzureCLIDeb | sudo bash
az login

# Create resource group
az group create --name clipper-rg --location eastus

# Deploy container
az container create \
  --resource-group clipper-rg \
  --name clipper-agent \
  --image clipper-agent:latest \
  --cpu 2 \
  --memory 4 \
  --ports 8000
```

#### 2. AKS Deployment
```bash
# Create AKS cluster
az aks create \
  --resource-group clipper-rg \
  --name clipper-cluster \
  --node-count 3 \
  --enable-addons monitoring \
  --generate-ssh-keys

# Get credentials
az aks get-credentials --resource-group clipper-rg --name clipper-cluster

# Deploy application
kubectl apply -f k8s/
```

## 🚀 Kubernetes Deployment

### Kubernetes Manifests

#### Namespace
```yaml
# k8s/namespace.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: clipper-agent
```

#### ConfigMap
```yaml
# k8s/configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: clipper-config
  namespace: clipper-agent
data:
  config.yaml: |
    processing:
      whisper_model_size: "base"
      max_clip_duration: 60
      min_clip_duration: 15
    storage:
      storage_path: "/app/storage"
      temp_path: "/app/temp"
    logging:
      log_level: "INFO"
```

#### Deployment
```yaml
# k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: clipper-agent
  namespace: clipper-agent
spec:
  replicas: 3
  selector:
    matchLabels:
      app: clipper-agent
  template:
    metadata:
      labels:
        app: clipper-agent
    spec:
      containers:
      - name: clipper-agent
        image: clipper-agent:latest
        ports:
        - containerPort: 8000
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "1000m"
        volumeMounts:
        - name: config
          mountPath: /app/config
        - name: storage
          mountPath: /app/storage
        env:
        - name: CLIPPER_LOG_LEVEL
          value: "INFO"
      volumes:
      - name: config
        configMap:
          name: clipper-config
      - name: storage
        persistentVolumeClaim:
          claimName: clipper-storage
```

#### Service
```yaml
# k8s/service.yaml
apiVersion: v1
kind: Service
metadata:
  name: clipper-agent-service
  namespace: clipper-agent
spec:
  selector:
    app: clipper-agent
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8000
  type: LoadBalancer
```

#### Persistent Volume Claim
```yaml
# k8s/pvc.yaml
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: clipper-storage
  namespace: clipper-agent
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 100Gi
```

### Deploy to Kubernetes
```bash
# Apply all manifests
kubectl apply -f k8s/

# Check deployment status
kubectl get pods -n clipper-agent
kubectl get services -n clipper-agent

# View logs
kubectl logs -f deployment/clipper-agent -n clipper-agent

# Scale deployment
kubectl scale deployment clipper-agent --replicas=5 -n clipper-agent
```

## 📊 Production Considerations

### Performance Optimization

#### Resource Allocation
```yaml
# Recommended resource limits
resources:
  requests:
    memory: "2Gi"
    cpu: "1000m"
  limits:
    memory: "4Gi"
    cpu: "2000m"
```

#### Horizontal Pod Autoscaler
```yaml
# k8s/hpa.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: clipper-agent-hpa
  namespace: clipper-agent
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: clipper-agent
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
```

### Security Configuration

#### Network Policies
```yaml
# k8s/network-policy.yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: clipper-agent-netpol
  namespace: clipper-agent
spec:
  podSelector:
    matchLabels:
      app: clipper-agent
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from:
    - namespaceSelector:
        matchLabels:
          name: ingress-nginx
    ports:
    - protocol: TCP
      port: 8000
  egress:
  - to: []
    ports:
    - protocol: TCP
      port: 443
    - protocol: TCP
      port: 80
```

#### Pod Security Policy
```yaml
# k8s/pod-security-policy.yaml
apiVersion: policy/v1beta1
kind: PodSecurityPolicy
metadata:
  name: clipper-agent-psp
spec:
  privileged: false
  allowPrivilegeEscalation: false
  requiredDropCapabilities:
    - ALL
  volumes:
    - 'configMap'
    - 'emptyDir'
    - 'projected'
    - 'secret'
    - 'downwardAPI'
    - 'persistentVolumeClaim'
  runAsUser:
    rule: 'MustRunAsNonRoot'
  seLinux:
    rule: 'RunAsAny'
  fsGroup:
    rule: 'RunAsAny'
```

### Monitoring and Logging

#### Prometheus Monitoring
```yaml
# k8s/servicemonitor.yaml
apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: clipper-agent-metrics
  namespace: clipper-agent
spec:
  selector:
    matchLabels:
      app: clipper-agent
  endpoints:
  - port: metrics
    interval: 30s
    path: /metrics
```

#### Logging Configuration
```yaml
# k8s/fluentd-config.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: fluentd-config
  namespace: clipper-agent
data:
  fluent.conf: |
    <source>
      @type tail
      path /var/log/containers/clipper-agent*.log
      pos_file /var/log/fluentd-clipper.log.pos
      tag clipper-agent.*
      format json
    </source>
    
    <match clipper-agent.**>
      @type elasticsearch
      host elasticsearch.logging.svc.cluster.local
      port 9200
      index_name clipper-agent
    </match>
```

## 🔧 Configuration Management

### Environment-Specific Configurations

#### Development
```yaml
# config/dev.yaml
processing:
  whisper_model_size: "tiny"
  max_clip_duration: 30
  
logging:
  log_level: "DEBUG"
  
storage:
  storage_path: "./dev_storage"
```

#### Staging
```yaml
# config/staging.yaml
processing:
  whisper_model_size: "base"
  max_clip_duration: 60
  
logging:
  log_level: "INFO"
  
storage:
  storage_path: "/app/staging_storage"
```

#### Production
```yaml
# config/prod.yaml
processing:
  whisper_model_size: "large"
  max_clip_duration: 120
  
logging:
  log_level: "WARNING"
  
storage:
  storage_path: "/app/storage"
```

### Secrets Management

#### Kubernetes Secrets
```bash
# Create secrets
kubectl create secret generic clipper-secrets \
  --from-literal=api-key=your-api-key \
  --from-literal=db-password=your-db-password \
  -n clipper-agent

# Use in deployment
env:
- name: API_KEY
  valueFrom:
    secretKeyRef:
      name: clipper-secrets
      key: api-key
```

#### HashiCorp Vault Integration
```yaml
# vault-config.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: vault-config
data:
  vault-addr: "https://vault.example.com"
  vault-role: "clipper-agent"
```

## 🚨 Disaster Recovery

### Backup Strategy

#### Database Backups
```bash
# PostgreSQL backup
kubectl exec -it postgres-pod -- pg_dump -U clipper clipper_agent > backup.sql

# Restore
kubectl exec -i postgres-pod -- psql -U clipper clipper_agent < backup.sql
```

#### Storage Backups
```bash
# Create storage backup
kubectl exec -it clipper-agent-pod -- tar -czf /tmp/storage-backup.tar.gz /app/storage

# Copy backup
kubectl cp clipper-agent-pod:/tmp/storage-backup.tar.gz ./storage-backup.tar.gz
```

### High Availability Setup

#### Multi-Region Deployment
```yaml
# Regional deployment with affinity rules
spec:
  replicas: 6
  template:
    spec:
      affinity:
        podAntiAffinity:
          preferredDuringSchedulingIgnoredDuringExecution:
          - weight: 100
            podAffinityTerm:
              labelSelector:
                matchExpressions:
                - key: app
                  operator: In
                  values:
                  - clipper-agent
              topologyKey: kubernetes.io/zone
```

## 📈 Scaling Strategies

### Vertical Scaling
```bash
# Increase resource limits
kubectl patch deployment clipper-agent -p '{"spec":{"template":{"spec":{"containers":[{"name":"clipper-agent","resources":{"limits":{"memory":"8Gi","cpu":"4000m"}}}]}}}}'
```

### Horizontal Scaling
```bash
# Manual scaling
kubectl scale deployment clipper-agent --replicas=10

# Auto-scaling with HPA
kubectl apply -f k8s/hpa.yaml
```

### Cost Optimization

#### Spot Instances (AWS)
```yaml
# Node group with spot instances
nodeGroups:
  - name: spot-workers
    instanceTypes: ["m5.large", "m5.xlarge", "m4.large"]
    spot: true
    minSize: 1
    maxSize: 10
    desiredCapacity: 3
```

#### Preemptible Instances (GCP)
```yaml
# Node pool with preemptible instances
gcloud container node-pools create preemptible-pool \
  --cluster=clipper-cluster \
  --preemptible \
  --num-nodes=3 \
  --machine-type=n1-standard-2
```

---

## 📞 Support

For deployment assistance:
- **Documentation**: Check our comprehensive guides
- **Community**: Join our Discord for real-time help
- **Professional Services**: Contact us for enterprise deployment support

**Next Steps**: After deployment, see [MONITORING_CHECKLIST.md](MONITORING_CHECKLIST.md) for production monitoring setup.