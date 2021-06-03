#version 330 core

// Ouput data
out vec4 FragColor;

in vec2 UV;

in vec3 Normal;
in vec3 FragPos;

uniform vec3 lightPos;
uniform vec3 viewPos;
uniform vec3 lightColor;
uniform vec3 objectColor;

uniform sampler2D tekstura01;

void main()
{
    // ambient
    float ambientStrength = 0.2;
    vec3 ambient = ambientStrength * lightColor;

    // diffuse
    vec3 norm = normalize(Normal);
    vec3 lightDir = normalize(lightPos - FragPos);
    //float diff = max(dot(norm, lightDir), 0.0);
    //float diff = clamp( abs(dot(norm, lightDir)), 0, 1);
    float diff = abs(dot(norm, lightDir));
    vec3 diffuse = diff * lightColor;

    // specular
    float specularStrength = 0.9;
    vec3 viewDir = normalize(viewPos - FragPos);
    vec3 reflectDir = reflect(-lightDir, norm);
    float cosAlpha = clamp( dot( viewDir,reflectDir ), 0, 1 );
    vec3 specular = specularStrength * pow(cosAlpha, 32) * lightColor;

    vec3 tekstura_diffuse = texture(tekstura01, UV).rgb;

    vec3 result = (ambient + diffuse + specular) * tekstura_diffuse/*objectColor*/;
    FragColor = vec4(result, 1.0);
}
