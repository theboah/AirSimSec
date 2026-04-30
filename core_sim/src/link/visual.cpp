// Copyright (C) Microsoft Corporation. 
// Copyright (C) 2025 IAMAI CONSULTING CORP

// MIT License. All rights reserved.

#include "core_sim/link/visual.hpp"

#include <memory>

#include "constant.hpp"
#include "core_sim/config_json.hpp"
#include "core_sim/error.hpp"
#include "core_sim/link/geometry/file_mesh.hpp"
#include "core_sim/link/geometry/unreal_mesh.hpp"
#include "core_sim/logger.hpp"
#include "core_sim/transforms/transform.hpp"
#include "geometry_impl.hpp"
#include "json.hpp"

namespace microsoft {
namespace projectairsim {

using json = nlohmann::json;

// -----------------------------------------------------------------------------
// Forward declarations

class Visual::Loader {
 public:
  Loader(Visual::Impl& impl);

  void Load(const json& json);

 private:
  void LoadOrigin(const json& json);

  void LoadHiddenInGame(const json& json);

  void LoadHiddenInSceneCapture(const json& json);

  void LoadGeometry(const json& json);

  void LoadMaterial(const json& json);

  std::string GetActorType(const json& json);

  Visual::Impl& impl_;
};

class Visual::Impl : public Component {
 public:
  Impl(const Logger& logger);

  void Load(ConfigJson config_json);

  const Transform& GetOrigin() const;

  const Geometry* GetGeometry() const;

  bool GetHiddenInGame() const;

  bool GetHiddenInSceneCapture() const;

  const Material& GetMaterial() const;

  operator const TransformTree::RefFrame&(void) const;

 private:
  friend class Visual::Loader;

  Visual::Loader loader_;
  Transform origin_;
  std::unique_ptr<Geometry> geo_;
  Material mat_;
  bool hidden_in_game_;
  bool hidden_in_scene_capture_;
  TransformTree::TransformRefFrame
      transformrefframe_;  // Inertial reference frame's transform tree node
};

// -----------------------------------------------------------------------------
// class Visual

Visual::Visual(const Logger& logger)
    : pimpl_(std::shared_ptr<Visual::Impl>(new Visual::Impl(logger))) {}

void Visual::Load(ConfigJson config_json) { return pimpl_->Load(config_json); }

bool Visual::IsLoaded() { return pimpl_->IsLoaded(); }

const Transform& Visual::GetOrigin() const { return pimpl_->GetOrigin(); }

const Geometry* Visual::GetGeometry() const { return pimpl_->GetGeometry(); }

bool Visual::GetHiddenInGame() const { return pimpl_->GetHiddenInGame(); }

bool Visual::GetHiddenInSceneCapture() const {
  return pimpl_->GetHiddenInSceneCapture();
}

const Material& Visual::GetMaterial() const { return pimpl_->GetMaterial(); }

Visual::operator TransformTree::RefFrame&(void) {
  // Call const version to avoid duplicating it with a non-cost version in the
  // impl--const_cast safe to do because this object is non-const in this call
  return (const_cast<TransformTree::RefFrame&>(
      pimpl_->operator const TransformTree::RefFrame&()));
}

Visual::operator const TransformTree::RefFrame&(void) const {
  return (pimpl_->operator const TransformTree::RefFrame&());
}

// -----------------------------------------------------------------------------
// class Visual::Impl

Visual::Impl::Impl(const Logger& logger)
    : Component(Constant::Component::visual, logger),
      loader_(*this),
      mat_(logger),
      transformrefframe_("Visual", &origin_) {
  hidden_in_game_ = false;
  hidden_in_scene_capture_ = false;
}

void Visual::Impl::Load(ConfigJson config_json) {
  json json = config_json;
  loader_.Load(json);
}

const Transform& Visual::Impl::GetOrigin() const { return origin_; }

const Geometry* Visual::Impl::GetGeometry() const { return geo_.get(); }

bool Visual::Impl::GetHiddenInGame() const { return hidden_in_game_; }

bool Visual::Impl::GetHiddenInSceneCapture() const {
  return hidden_in_scene_capture_;
}

const Material& Visual::Impl::GetMaterial() const { return mat_; }

Visual::Impl::operator const TransformTree::RefFrame&(void) const {
  return (transformrefframe_);
}

// -----------------------------------------------------------------------------
// class Visual::Loader

Visual::Loader::Loader(Visual::Impl& impl) : impl_(impl) {}

void Visual::Loader::Load(const json& json) {
  LoadOrigin(json);
  LoadHiddenInGame(json);
  LoadHiddenInSceneCapture(json);
  LoadGeometry(json);
  LoadMaterial(json);

  impl_.is_loaded_ = true;
}

void Visual::Loader::LoadOrigin(const json& json) {
  impl_.logger_.LogVerbose(impl_.name_, "Loading 'origin'.");

  impl_.origin_ = JsonUtils::GetTransform(json, Constant::Config::origin);

  impl_.logger_.LogVerbose(impl_.name_, "'origin' loaded.");
}

void Visual::Loader::LoadHiddenInGame(const json& json) {
  impl_.logger_.LogVerbose(impl_.name_, "Loading 'hidden_in_game'.");

  impl_.hidden_in_game_ =
      JsonUtils::GetBoolean(json, Constant::Config::hidden_in_game, false);

  impl_.logger_.LogVerbose(impl_.name_, "'hidden_in_game' loaded.");
}

void Visual::Loader::LoadHiddenInSceneCapture(const json& json) {
  impl_.logger_.LogVerbose(impl_.name_, "Loading 'hidden_in_scene_capture'.");

  impl_.hidden_in_scene_capture_ = JsonUtils::GetBoolean(
      json, Constant::Config::hidden_in_scene_capture, false);

  impl_.logger_.LogVerbose(impl_.name_, "'hidden_in_scene_capture' loaded.");
}

void Visual::Loader::LoadGeometry(const json& json) {
  impl_.logger_.LogVerbose(impl_.name_, "Loading 'geometry'.");

  auto geometry_json =
      JsonUtils::GetJsonObject(json, Constant::Config::geometry);
  if (JsonUtils::IsEmpty(geometry_json)) {
    impl_.logger_.LogWarning(impl_.name_, "'geometry' missing or empty.");
    return;
  }

  auto type = GetActorType(geometry_json);

  if (type == Constant::Config::file_mesh) {
    auto mesh = new FileMesh(impl_.logger_);
    mesh->Load(geometry_json);
    impl_.geo_.reset(mesh);
  } else if (type == Constant::Config::unreal_mesh) {
    auto mesh = new UnrealMesh(impl_.logger_);
    mesh->Load(geometry_json);
    impl_.geo_.reset(mesh);
  } else {
    impl_.logger_.LogError(impl_.name_, "Invalid geometry type '%s'.",
                           type.c_str());
    throw Error("Invalid geometry type.");
  }

  impl_.logger_.LogVerbose(impl_.name_, "'geometry' loaded.");
}

void Visual::Loader::LoadMaterial(const json& json) {
  auto material_json =
      JsonUtils::GetJsonObject(json, Constant::Config::material);
  if (!JsonUtils::IsEmpty(material_json)) {
    impl_.logger_.LogVerbose(impl_.name_, "Loading 'material'.");

    impl_.mat_.Load(material_json);
    impl_.logger_.LogVerbose(impl_.name_, "'material' loaded.");
  }
}

std::string Visual::Loader::GetActorType(const json& json) {
  return JsonUtils::GetIdentifier(json, Constant::Config::type);
}

}  // namespace projectairsim
}  // namespace microsoft
