#enviar algo para a VM:
gcloud compute scp PATH_TO_FILE  --zone INSTANCE_ZONE INSTANCE_NAME:~
#acessar a VM:
gcloud compute ssh --zone INSTANCE_ZONE INSTANCE_NAME

#nome usuário: sa_109649273287045369425 (o número é o client_id da conta de servico)