DOCKER_CMD="${DOCKER_CMD:-podman}"

rm volpe_img.tar
$DOCKER_CMD build -t volpe_grpc_test .
$DOCKER_CMD save -o volpe_img.tar volpe_grpc_test
