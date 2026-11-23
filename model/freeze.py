def apply_freeze_strategy(model, architecture, strategy):
    """Freeze the requested early backbone components or train only the classifier."""
    strategies = {"none", "classifier_only", "stem", "first_stage", "first_two_stages"}
    if strategy not in strategies:
        raise ValueError(
            f"Unsupported freeze strategy {strategy!r}; expected one of {sorted(strategies)}."
        )
    for parameter in model.parameters():
        parameter.requires_grad = True
    if strategy == "none":
        return

    backbone = model.model
    classifier = getattr(backbone, "classifier", None)
    if classifier is None and hasattr(backbone, "fc"):
        classifier = backbone.fc
    if classifier is None:
        raise ValueError(f"Could not identify the classifier for {architecture}.")

    if strategy == "classifier_only":
        for parameter in model.parameters():
            parameter.requires_grad = False
        for parameter in classifier.parameters():
            parameter.requires_grad = True
        return

    if architecture.startswith("ResNet"):
        frozen_modules = [backbone.conv1, backbone.bn1]
        if strategy in {"first_stage", "first_two_stages"}:
            frozen_modules.append(backbone.layer1)
        if strategy == "first_two_stages":
            frozen_modules.append(backbone.layer2)
    elif architecture == "SqueezeNet":
        features = backbone.features
        frozen_modules = [features[0]]
        if strategy in {"first_stage", "first_two_stages"}:
            frozen_modules.extend(features[3:5])
        if strategy == "first_two_stages":
            frozen_modules.extend(features[6:9])
    elif architecture == "EfficientNetB0":
        features = backbone.features
        frozen_modules = [features[0]]
        if strategy in {"first_stage", "first_two_stages"}:
            frozen_modules.append(features[1])
        if strategy == "first_two_stages":
            frozen_modules.append(features[2])
    else:
        raise ValueError(f"Freeze strategies are not configured for {architecture}.")

    for module in frozen_modules:
        for parameter in module.parameters():
            parameter.requires_grad = False
